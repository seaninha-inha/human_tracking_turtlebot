#!/usr/bin/env python3

import queue
import threading

import cv2
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
        # 이 거리까지 다가가 정지
        self.target_distance      = rospy.get_param("~target_distance",  1.2)
        self.depth_window_px      = rospy.get_param("~depth_window_px",  15)
        self.center_deadband_frac = rospy.get_param("~center_deadband_frac", 0.04)  # 폭 대비 비율
        self.min_valid_depth      = rospy.get_param("~min_valid_depth",  0.2)
        self.max_valid_depth      = rospy.get_param("~max_valid_depth",  15.0)
        self.linear_gain          = rospy.get_param("~linear_gain",   0.5)
        self.angular_gain         = rospy.get_param("~angular_gain",  1.2)
        self.max_linear_speed     = rospy.get_param("~max_linear_speed",  0.8)
        self.max_angular_speed    = rospy.get_param("~max_angular_speed", 1.2)
        self.detection_timeout    = rospy.Duration(rospy.get_param("~detection_timeout", 0.7))
        self.control_rate         = rospy.get_param("~control_rate", 15.0)
        self.sync_slop            = rospy.get_param("~sync_slop", 0.05)  # 동기화 허용 시간차(초)

        # ── 상태 변수 ─────────────────────────────────────────────
        self.latest_target      = None
        self.last_detection_time = rospy.Time(0)

        self.target_track_id = None # None이면 추적 대상 없음, 숫자면 해당 ID 추적 중
        self.current_boxes = [] # 현재 프레임의 모든 박스 정보 저장
        self.window_name   = "YOLO target select (click a person)"

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
        rospy.loginfo("Click a person in the window to start following. (r=reset, ESC=quit)")

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

    # ── YOLO 추론 워커 (+ GUI) ─────────────────────────────
    def _infer_worker(self):
        window_ready = False
        while not rospy.is_shutdown():
            try:
                frame, depth, encoding = self._infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.model is None:
                continue

            result = self.model.track(frame, persist=True, conf=self.confidence_threshold, classes=[0], verbose=False)[0]
            boxes  = self._person_boxes(result)
            self.current_boxes = boxes

            target_box = None
            if self.target_track_id is not None:
                for box in boxes:
                    if box[4] == self.target_track_id:
                        target_box = box
                        break
            
            if target_box is None:
                with self.lock:
                    self.latest_target = None
            else:
                x1, y1, x2, y2, _tid = target_box
                img_w = frame.shape[1]
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
                    }
                    self.last_detection_time = rospy.Time.now()

            self._draw(frame, boxes)
            if not window_ready:
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(self.window_name, self._on_mouse)
                window_ready = True
            cv2.imshow(self.window_name, frame)
        
            key = cv2.waitKey(1) & 0xFF
            # 'r' 키: 타겟 리셋, ESC 키: 종료
            if key == ord('r'):
                self.target_track_id = None
                rospy.loginfo("Target reset; click a person again.")
            elif key == 27:  # ESC
                rospy.signal_shutdown("user quit")
 
    # ── 마우스 클릭: 클릭 지점을 포함하는 박스의 track_id 선택 ────
    def _on_mouse(self, event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, tid in self.current_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.target_track_id = tid
                rospy.loginfo("Target selected by click: track_id=%d", tid)
                return
 
    def _person_boxes(self, result):
        out = []
        if result.boxes is None or result.boxes.id is None:
            return out
        ids  = result.boxes.id.int().tolist()
        xyxy = result.boxes.xyxy.tolist()
        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            out.append((int(x1), int(y1), int(x2), int(y2), int(tid)))
        return out
 
    def _draw(self, frame, boxes):
        for x1, y1, x2, y2, tid in boxes:
            selected = (tid == self.target_track_id)
            color = (0, 255, 0) if selected else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if selected else 1)
            label = ("TARGET " if selected else "") + ("ID:%d" % tid)
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if self.target_track_id is None:
            cv2.putText(frame, "Click a person to follow", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # 깊이 샘플링: 타겟까지의 거리 계산
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

    #  제어 루프 (P제어: 각속도 P, 선속도 P)
    def control_callback(self, _event):
        now = rospy.Time.now()
        cmd = Twist()

        with self.lock:
            target = self.latest_target
            age    = now - self.last_detection_time

        # 탐색 대상 없거나 타임아웃
        if target is None or age > self.detection_timeout:
            self.cmd_pub.publish(cmd)
            return

        # 각속도: 정규화된 center_error에 비례
        center_error_norm = target["center_error_norm"]
        if abs(center_error_norm) > target["center_deadband"]:
            cmd.angular.z = self.clamp(-center_error_norm * self.angular_gain,
                                       -self.max_angular_speed,
                                       self.max_angular_speed)

        # 선속도: (현재거리 - 목표거리에 비례, 목표거리 이내면 정지)
        distance = target["distance"]   # 디버깅
        if distance is not None:
            distance_error = distance - self.target_distance
            
            if distance_error > 0.0:
                cmd.linear.x = self.clamp(distance_error * self.linear_gain, 0.0, self.max_linear_speed)

        self.cmd_pub.publish(cmd)

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    rospy.init_node("yolo_depth_follower")
    YoloDepthFollower()
    rospy.spin()