#!/usr/bin/env python3

import queue
import threading

import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


class YoloDepthFollower:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # ── 파라미터 ──────────────────────────────────────────────
        self.rgb_topic   = rospy.get_param("~rgb_topic",   "/camera/rgb/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_raw")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.model_path  = rospy.get_param("~model_path",  "yolov8n.pt")

        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.45)
        self.target_distance      = rospy.get_param("~target_distance",  1.2)
        self.stop_distance        = rospy.get_param("~stop_distance",    0.8)
        self.depth_window_px      = rospy.get_param("~depth_window_px",  15)
        self.center_deadband_frac = rospy.get_param("~center_deadband_frac", 0.04)  # 폭 대비 비율
        self.min_valid_depth      = rospy.get_param("~min_valid_depth",  0.2)
        self.max_valid_depth      = rospy.get_param("~max_valid_depth",  8.0)
        self.linear_gain          = rospy.get_param("~linear_gain",   0.5)
        self.linear_d_gain        = rospy.get_param("~linear_d_gain", 0.2)  # 추가: D항
        self.angular_gain         = rospy.get_param("~angular_gain",  1.2)
        self.max_linear_speed     = rospy.get_param("~max_linear_speed",  0.8)
        self.max_angular_speed    = rospy.get_param("~max_angular_speed", 1.2)
        self.detection_timeout    = rospy.Duration(rospy.get_param("~detection_timeout", 0.7))
        self.search_when_lost     = rospy.get_param("~search_when_lost", False)
        self.search_angular_speed = rospy.get_param("~search_angular_speed", 0.25)
        self.control_rate         = rospy.get_param("~control_rate", 15.0)
        self.sync_slop            = rospy.get_param("~sync_slop", 0.05)  # 동기화 허용 시간차(초)

        # ── 상태 변수 ─────────────────────────────────────────────
        self.latest_target      = None
        self.last_detection_time = rospy.Time(0)
        self.prev_distance_error = 0.0          # D항용
        self.prev_control_time   = rospy.Time(0)

        # ── YOLO 추론 전용 스레드 ─────────────────────────────────
        self._infer_queue = queue.Queue(maxsize=1)   # 최신 1프레임만 유지
        self._infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        self._infer_thread.start()

        self.model = self._load_model()

        # ── ROS I/O ───────────────────────────────────────────────
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)

        # RGB + Depth 시간 동기화 (slop 이내 프레임 쌍만 처리)
        rgb_sub   = message_filters.Subscriber(self.rgb_topic,   Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=self.sync_slop
        )
        self.ts.registerCallback(self.synced_callback)

        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate), self.control_callback
        )
        rospy.on_shutdown(self.stop_robot)
        rospy.loginfo("YOLO depth follower ready  rgb=%s  depth=%s  cmd_vel=%s",
                      self.rgb_topic, self.depth_topic, self.cmd_vel_topic)

    # ── 모델 로드 ─────────────────────────────────────────────────
    def _load_model(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            rospy.loginfo("Loaded YOLO model: %s", self.model_path)
            return model
        except Exception as exc:
            rospy.logerr("Failed to load YOLO model '%s': %s", self.model_path, exc)
            return None

    # RGB + Depth가 시간적으로 맞는 쌍이 도착하면 호출됨
    def synced_callback(self, rgb_msg, depth_msg):
        if self.model is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "CvBridge error: %s", exc)
            return

        # 추론 큐에 넣기 (꽉 차 있으면 오래된 것을 버리고 최신만 유지)
        try:
            self._infer_queue.get_nowait()
        except queue.Empty:
            pass
        self._infer_queue.put_nowait((frame, depth, depth_msg.encoding))

    # ── YOLO 추론 워커 (별도 스레드) ─────────────────────────────
    def _infer_worker(self):
        while not rospy.is_shutdown():
            try:
                frame, depth, encoding = self._infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.model is None:
                continue

            result = self.model(frame, verbose=False, conf=self.confidence_threshold)[0]
            box    = self.select_person_box(result)

            if box is None:
                with self.lock:
                    self.latest_target = None
                continue

            x1, y1, x2, y2, confidence = box
            img_h, img_w = frame.shape[:2]
            center_x = int((x1 + x2) * 0.5)
            center_y = int((y1 + y2) * 0.5)

            distance = self._depth_at(depth, encoding, center_x, center_y)

            # center_error를 [-1, 1] 정규화 (이미지 폭에 독립적)
            center_error_norm = (center_x - img_w * 0.5) / (img_w * 0.5)

            with self.lock:
                self.latest_target = {
                    "center_error_norm": center_error_norm,
                    "center_deadband":   self.center_deadband_frac,  # 동일 기준
                    "distance":          distance,
                    "confidence":        confidence,
                }
                self.last_detection_time = rospy.Time.now()

    # ── 깊이 샘플링 ───────────────────────────────────────────────
    def _depth_at(self, depth, encoding, cx, cy):
        half = max(1, int(self.depth_window_px * 0.5))
        x1 = max(0, cx - half);  x2 = min(depth.shape[1], cx + half + 1)
        y1 = max(0, cy - half);  y2 = min(depth.shape[0], cy + half + 1)

        patch = depth[y1:y2, x1:x2].astype(np.float32)
        if encoding == "16UC1":
            patch *= 0.001

        valid = patch[np.isfinite(patch)]
        valid = valid[(valid >= self.min_valid_depth) & (valid <= self.max_valid_depth)]
        return float(np.median(valid)) if valid.size > 0 else None

    # ── 가장 큰 사람 박스 선택 ────────────────────────────────────
    def select_person_box(self, result):
        if result.boxes is None:
            return None
        best_box, best_area = None, 0.0
        for box in result.boxes:
            if int(box.cls[0]) != 0 or float(box.conf[0]) < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best_box  = (x1, y1, x2, y2, float(box.conf[0]))
        return best_box

    # ── 제어 루프 ─────────────────────────────────────────────────
    def control_callback(self, _event):
        now = rospy.Time.now()
        cmd = Twist()

        with self.lock:
            target = self.latest_target
            age    = now - self.last_detection_time

        # 탐색 대상 없거나 타임아웃
        if target is None or age > self.detection_timeout:
            if self.search_when_lost:
                cmd.angular.z = self.search_angular_speed
            self.cmd_pub.publish(cmd)
            self.prev_distance_error = 0.0
            return

        # ── 각속도: 정규화된 center_error 사용 ──────────────────
        center_error_norm = target["center_error_norm"]
        if abs(center_error_norm) > target["center_deadband"]:
            raw_angular = -center_error_norm * self.angular_gain
            cmd.angular.z = self.clamp(raw_angular,
                                       -self.max_angular_speed,
                                       self.max_angular_speed)

        # ── 선속도: 거리 오차 PD 제어 ───────────────────────────
        distance = target["distance"]
        if distance is not None:
            dt = (now - self.prev_control_time).to_sec() if self.prev_control_time != rospy.Time(0) else 0.1
            dt = max(dt, 1e-3)

            distance_error = distance - self.target_distance

            # D항: 오차 변화율
            d_term = self.linear_d_gain * (distance_error - self.prev_distance_error) / dt
            self.prev_distance_error = distance_error

            if distance <= self.stop_distance:
                # 너무 가까우면 선속도만 0, 각속도는 유지
                cmd.linear.x = 0.0
            elif distance_error > 0.0:
                raw_linear = distance_error * self.linear_gain + d_term
                cmd.linear.x = self.clamp(raw_linear, 0.0, self.max_linear_speed)

        self.prev_control_time = now
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