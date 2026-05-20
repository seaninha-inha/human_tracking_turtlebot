#!/usr/bin/env python3

import math
import threading

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

        self.model = self._load_model()
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)
        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        self.control_timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_callback)

        rospy.on_shutdown(self.stop_robot)
        rospy.loginfo("YOLO depth follower started: rgb=%s depth=%s cmd_vel=%s",
                      self.rgb_topic, self.depth_topic, self.cmd_vel_topic)

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

        result = self.model(frame, verbose=False, conf=self.confidence_threshold)[0]
        box = self.select_person_box(result)
        if box is None:
            with self.lock:
                self.latest_target = None
            return

        x1, y1, x2, y2, confidence = box
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
            }
            self.last_detection_time = rospy.Time.now()

    def select_person_box(self, result):
        if result.boxes is None:
            return None

        best_box = None
        best_area = 0.0
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id != 0 or confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2, confidence)
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


if __name__ == "__main__":
    rospy.init_node("yolo_depth_follower")
    YoloDepthFollower()
    rospy.spin()