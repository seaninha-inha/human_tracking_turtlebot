#!/usr/bin/env python3

import threading

import rospy
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from ultralytics import YOLO

class YoloDetector:

    def __init__(self):
        rospy.init_node("yolo_detector")

        # Parameters
        self.model_path = rospy.get_param(
            "~model_path",
            "yolov8n.pt"
        )

        self.rgb_topic = rospy.get_param(
            "~rgb_topic",
            "/camera/rgb/image_raw"
        )

        self.output_topic = rospy.get_param(
            "~output_topic",
            "/yolo_detection_image"
        )

        self.conf_threshold = rospy.get_param(
            "~confidence_threshold",
            0.45
        )
        self.process_rate = rospy.get_param(
            "~process_rate",
            5.0
        )

        # YOLO model
        self.model = YOLO(self.model_path)
        # CV Bridge
        self.bridge = CvBridge()
        # Latest frame buffer for async processing
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.processing = False
        # Subscriber
        self.image_sub = rospy.Subscriber(
            self.rgb_topic,
            Image,
            self.image_callback,
            queue_size=1
        )
        # Publisher
        self.image_pub = rospy.Publisher(
            self.output_topic,
            Image,
            queue_size=1
        )
        self.process_timer = rospy.Timer(
            rospy.Duration(1.0 / self.process_rate),
            self.process_frame_callback
        )

        rospy.loginfo("YOLO detector started.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            rospy.logerr("CV Bridge Error: %s", e)
            return

        with self.frame_lock:
            self.latest_frame = frame

    def process_frame_callback(self, _event):
        with self.frame_lock:
            if self.processing or self.latest_frame is None:
                return
            frame = self.latest_frame.copy()
            self.processing = True

        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                verbose=False
            )

            annotated_frame = frame.copy()

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])

                    # person class only
                    if cls_id != 0:
                        continue

                    track_id = int(box.id[0]) if box.id is not None else -1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    label = f"ID:{track_id}"

                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            ros_image = self.bridge.cv2_to_imgmsg(
                annotated_frame,
                encoding="bgr8"
            )
            self.image_pub.publish(ros_image)

        except Exception as e:
            rospy.logerr("Publish Error: %s", e)
        finally:
            with self.frame_lock:
                self.processing = False


# Main
if __name__ == "__main__":
    detector = YoloDetector()
    rospy.spin()