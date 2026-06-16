#!/usr/bin/env python3

import os
import math
import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


class YoloDepthFollower:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.rgb_topic = rospy.get_param("~rgb_topic", "/camera/rgb/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_raw")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.model_path = rospy.get_param("~model_path", "yolov8n.pt")
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "yolo_detection_image")
        self.enable_gui = rospy.get_param("~enable_gui", False)

        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.45)
        self.target_distance = rospy.get_param("~target_distance", 1.2)
        self.stop_distance = rospy.get_param("~stop_distance", 0.8)
        self.depth_window_px = rospy.get_param("~depth_window_px", 15)
        self.center_deadband_px = rospy.get_param("~center_deadband_px", 25)
        self.min_valid_depth = rospy.get_param("~min_valid_depth", 0.2)
        self.max_valid_depth = rospy.get_param("~max_valid_depth", 8.0)
        self.linear_gain = rospy.get_param("~linear_gain", 0.35)
        self.angular_gain = rospy.get_param("~angular_gain", 1.2)
        self.max_linear_speed = rospy.get_param("~max_linear_speed", 0.22)
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 1.2)
        self.detection_timeout = rospy.Duration(rospy.get_param("~detection_timeout", 0.7))
        self.search_when_lost = rospy.get_param("~search_when_lost", False)
        self.search_angular_speed = rospy.get_param("~search_angular_speed", 0.25)
        self.control_rate = rospy.get_param("~control_rate", 10.0)

        self.latest_depth = None
        self.latest_depth_encoding = None
        self.latest_target = None
        self.last_detection_time = rospy.Time(0)
        
        # Target lock variables
        self.target_track_id = None  # None = waiting for initial auto selection
        self.all_person_boxes = []  # List of (x1, y1, x2, y2, track_id)

        # Visualization setup
        self.window_name = "YOLO Detection - Auto target lock"

        self.model = self._load_model()
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)
        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        self.control_timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_callback)

        # Setup visualization window only when GUI is available.
        if self.enable_gui:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        rospy.on_shutdown(self.stop_robot)
        rospy.loginfo("YOLO depth follower started: rgb=%s depth=%s cmd_vel=%s",
                      self.rgb_topic, self.depth_topic, self.cmd_vel_topic)
        if self.enable_gui:
            rospy.loginfo("Auto-selects the nearest person on first detection. Press 'r' to reset target.")
        else:
            rospy.loginfo("GUI is disabled; debug images are published on %s", self.debug_image_topic)

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            rospy.loginfo("Loaded YOLO model: %s", self.model_path)
            return model
        except Exception as exc:
            rospy.logerr("Failed to load ultralytics YOLO model '%s': %s", self.model_path, exc)
            rospy.logerr("Install ultralytics or set _model_path to an available YOLO model file.")
            return None

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "Failed to convert depth image: %s", exc)
            return

        with self.lock:
            self.latest_depth = depth
            self.latest_depth_encoding = msg.encoding

    def rgb_callback(self, msg):
        if self.model is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "Failed to convert RGB image: %s", exc)
            return

        # YOLO tracking
        results = self.model.track(frame, persist=True, conf=self.confidence_threshold, verbose=False)

        # Collect all detected persons
        self.all_person_boxes = self.get_all_person_boxes(results[0])
        
        # Select target based on lock status.
        box = None
        if self.target_track_id is None:
            box = self.select_nearest_person_box(results[0])
            if box is not None:
                with self.lock:
                    self.target_track_id = box[5]
                rospy.loginfo("Target auto-locked to nearest person: track_id=%d", box[5])
        else:
            box = self.get_box_by_track_id(results[0], self.target_track_id)
            if box is None:
                rospy.logwarn_throttle(2.0, "Target track_id=%s not found in current frame", self.target_track_id)
        
        # Visualize frame with detections
        display_frame = frame.copy()
        self._draw_detections(display_frame)
        try:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(display_frame, encoding="bgr8"))
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "Failed to publish debug image: %s", exc)
        if self.enable_gui:
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # Press 'r' to reset target
                with self.lock:
                    self.target_track_id = None
                rospy.loginfo("Target reset; nearest person will be selected again")
        
        if box is None:
            with self.lock:
                self.latest_target = None
            return

        x1, y1, x2, y2, confidence, track_id = box
        center_x = int((x1 + x2) * 0.5)
        center_y = int((y1 + y2) * 0.5)
        distance = self.depth_at(center_x, center_y)
        image_center_x = frame.shape[1] * 0.5
        center_error = center_x - image_center_x

        with self.lock:
            self.latest_target = {
                "center_error": center_error,
                "distance": distance,
                "confidence": confidence,
                "track_id": track_id,
            }
            self.last_detection_time = rospy.Time.now()

    def select_nearest_person_box(self, result):
        if result.boxes is None:
            return None

        best_box = None
        best_distance = None
        best_area = 0.0
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id != 0:
                continue

            confidence = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            center_x = int((x1 + x2) * 0.5)
            center_y = int((y1 + y2) * 0.5)
            distance = self.depth_at(center_x, center_y)

            if distance is None:
                continue

            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if best_distance is None or distance < best_distance or (distance == best_distance and area > best_area):
                best_distance = distance
                best_area = area
                best_box = (x1, y1, x2, y2, confidence, track_id)

        return best_box

    def depth_at(self, center_x, center_y):
        with self.lock:
            if self.latest_depth is None:
                return None
            depth = self.latest_depth.copy()
            encoding = self.latest_depth_encoding

        half = max(1, int(self.depth_window_px * 0.5))
        x1 = max(0, center_x - half)
        x2 = min(depth.shape[1], center_x + half + 1)
        y1 = max(0, center_y - half)
        y2 = min(depth.shape[0], center_y + half + 1)
        patch = depth[y1:y2, x1:x2].astype(np.float32)

        if encoding == "16UC1":
            patch *= 0.001

        valid = patch[np.isfinite(patch)]
        valid = valid[(valid >= self.min_valid_depth) & (valid <= self.max_valid_depth)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def get_all_person_boxes(self, result):
        """Get all detected person bboxes as list of (x1, y1, x2, y2, track_id)"""
        boxes = []
        if result.boxes is None:
            return boxes
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id != 0:  # Only persons (class 0)
                continue
            
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            boxes.append((x1, y1, x2, y2, track_id))
        
        return boxes

    def get_box_by_track_id(self, result, target_id):
        """Get bbox for specific track_id"""
        if result.boxes is None:
            return None
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id != 0:
                continue
            
            track_id = int(box.id[0]) if box.id is not None else -1
            if track_id == target_id:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                return (x1, y1, x2, y2, confidence, track_id)
        
        return None

    def _draw_detections(self, frame):
        """Draw bboxes on frame"""
        for x1, y1, x2, y2, track_id in self.all_person_boxes:
            if track_id == self.target_track_id:
                # Target person: green box
                color = (0, 255, 0)
                thickness = 3
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"TARGET ID:{track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                # Other persons: blue box
                color = (255, 0, 0)
                thickness = 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        
        # Display instruction
        if self.target_track_id is None:
            cv2.putText(frame, "Click on a person to select target", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    def control_callback(self, _event):
        cmd = Twist()
        now = rospy.Time.now()

        with self.lock:
            target = self.latest_target
            age = now - self.last_detection_time

        if target is None or age > self.detection_timeout:
            if self.search_when_lost:
                cmd.angular.z = self.search_angular_speed
            self.cmd_pub.publish(cmd)
            return

        center_error = target["center_error"]
        distance = target["distance"]

        if abs(center_error) > self.center_deadband_px:
            cmd.angular.z = -self.clamp(center_error / 320.0 * self.angular_gain,
                                        -self.max_angular_speed,
                                        self.max_angular_speed)

        if distance is None:
            self.cmd_pub.publish(cmd)
            return

        if distance <= self.stop_distance:
            self.cmd_pub.publish(Twist())
            return

        distance_error = distance - self.target_distance
        if distance_error > 0.0:
            cmd.linear.x = self.clamp(distance_error * self.linear_gain, 0.0, self.max_linear_speed)

        self.cmd_pub.publish(cmd)

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        if self.enable_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node("yolo_depth_follower")
    YoloDepthFollower()
    rospy.spin()
