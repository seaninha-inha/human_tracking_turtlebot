#!/usr/bin/env python3

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

        # YOLO model
        self.model = YOLO(self.model_path)
        # CV Bridge
        self.bridge = CvBridge()
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

        rospy.loginfo("YOLO detector started.")

    # Image Callback
    def image_callback(self, msg):
        try:
            # ROS Image -> OpenCV
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            rospy.logerr("CV Bridge Error: %s", e)
            return

        # YOLO inference
        results = self.model(frame)
        # Draw detections
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # person class only
                if cls_id != 0:
                    continue

                # confidence threshold
                if conf < self.conf_threshold:
                    continue

                # bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # draw rectangle
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # label text
                label = f"person {conf:.2f}"

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Publish result image
        try:
            ros_image = self.bridge.cv2_to_imgmsg(
                annotated_frame,
                encoding="bgr8"
            )
            self.image_pub.publish(ros_image)

        except Exception as e:
            rospy.logerr("Publish Error: %s", e)


# Main
if __name__ == "__main__":
    detector = YoloDetector()
    rospy.spin()