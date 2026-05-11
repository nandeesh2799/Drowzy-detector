from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import time

import cv2
import numpy as np


# Face mesh indices.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [61, 81, 13, 311, 291, 178, 14, 402]
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Approximate camera model points for solvePnP.
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),       # nose tip
        (0.0, -63.6, -12.5),   # chin
        (-43.3, 32.7, -26.0),  # left eye outer corner
        (43.3, 32.7, -26.0),   # right eye outer corner
        (-28.9, -28.9, -24.1), # left mouth corner
        (28.9, -28.9, -24.1),  # right mouth corner
    ],
    dtype=np.float64,
)


@dataclass
class FrameSignals:
    timestamp: float
    face_detected: bool
    multiple_faces: bool
    ear: float
    mar: float
    blink_rate: float
    blink_detected: bool
    head_pitch: float
    head_yaw: float
    gaze_away: bool
    low_light: bool
    glasses_fallback: bool
    face_box_area: float


class FeatureExtractor:
    def __init__(self, blink_window_seconds: float = 60.0) -> None:
        self.blink_window_seconds = blink_window_seconds
        self._prev_eye_closed = False
        self._blink_events: List[float] = []

    @staticmethod
    def _dist(a: Sequence[float], b: Sequence[float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _ear(self, lm_xy: Dict[int, Tuple[float, float]], idx: List[int]) -> float:
        p = [lm_xy[i] for i in idx]
        horizontal = self._dist(p[0], p[3])
        if horizontal <= 1e-6:
            return 0.0
        return (self._dist(p[1], p[5]) + self._dist(p[2], p[4])) / (2.0 * horizontal)

    def _mar(self, lm_xy: Dict[int, Tuple[float, float]]) -> float:
        p = [lm_xy[i] for i in MOUTH]
        horizontal = self._dist(p[0], p[4])
        if horizontal <= 1e-6:
            return 0.0
        vertical = (
            self._dist(p[1], p[7]) + self._dist(p[2], p[6]) + self._dist(p[3], p[5])
        ) / 3.0
        return vertical / horizontal

    def _blink_rate(self, now: float, eye_closed: bool) -> Tuple[float, bool]:
        blink_detected = False
        if eye_closed and not self._prev_eye_closed:
            self._blink_events.append(now)
            blink_detected = True
        self._prev_eye_closed = eye_closed

        cutoff = now - self.blink_window_seconds
        self._blink_events = [t for t in self._blink_events if t >= cutoff]
        per_minute = len(self._blink_events) * (60.0 / self.blink_window_seconds)
        return per_minute, blink_detected

    @staticmethod
    def _face_area(lm_px: Dict[int, Tuple[float, float]]) -> float:
        xs = [p[0] for p in lm_px.values()]
        ys = [p[1] for p in lm_px.values()]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return max(0.0, width * height)

    @staticmethod
    def _extract_landmarks(
        landmarks,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
        lm_xy: Dict[int, Tuple[float, float]] = {}
        lm_px: Dict[int, Tuple[float, float]] = {}
        for i, p in enumerate(landmarks):
            x = float(p.x)
            y = float(p.y)
            lm_xy[i] = (x, y)
            lm_px[i] = (x * frame_w, y * frame_h)
        return lm_xy, lm_px

    @staticmethod
    def _head_pose(
        lm_px: Dict[int, Tuple[float, float]],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[float, float]:
        frame_h, frame_w = frame_shape[:2]
        image_points = np.array(
            [
                lm_px[NOSE_TIP],
                lm_px[CHIN],
                lm_px[LEFT_EYE_OUTER],
                lm_px[RIGHT_EYE_OUTER],
                lm_px[MOUTH_LEFT],
                lm_px[MOUTH_RIGHT],
            ],
            dtype=np.float64,
        )

        focal_length = float(frame_w)
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, _ = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
        # Positive pitch -> looking down.
        # Yaw sign is kept native to match camera coordinate output.
        pitch = -float(euler[0, 0])
        yaw = float(euler[1, 0])
        return pitch, yaw

    def select_main_face(self, multi_face_landmarks, frame_w: int, frame_h: int):
        if not multi_face_landmarks:
            return None, False, 0.0

        best_face = None
        best_area = -1.0
        for face in multi_face_landmarks:
            _, lm_px = self._extract_landmarks(face.landmark, frame_w, frame_h)
            area = self._face_area(lm_px)
            if area > best_area:
                best_face = face
                best_area = area
        return best_face, len(multi_face_landmarks) > 1, best_area

    def compute_signals(
        self,
        frame: np.ndarray,
        multi_face_landmarks,
        calibrated_ear_threshold: float,
        gaze_yaw_threshold: float,
    ) -> FrameSignals:
        now = time.time()
        frame_h, frame_w = frame.shape[:2]
        brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        low_light = brightness < 70.0

        if not multi_face_landmarks:
            return FrameSignals(
                timestamp=now,
                face_detected=False,
                multiple_faces=False,
                ear=0.0,
                mar=0.0,
                blink_rate=0.0,
                blink_detected=False,
                head_pitch=0.0,
                head_yaw=0.0,
                gaze_away=False,
                low_light=low_light,
                glasses_fallback=False,
                face_box_area=0.0,
            )

        face, multiple_faces, area = self.select_main_face(multi_face_landmarks, frame_w, frame_h)
        lm_xy, lm_px = self._extract_landmarks(face.landmark, frame_w, frame_h)

        left_ear = self._ear(lm_xy, LEFT_EYE)
        right_ear = self._ear(lm_xy, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        mar = self._mar(lm_xy)

        eye_closed = ear < calibrated_ear_threshold
        blink_rate, blink_detected = self._blink_rate(now, eye_closed)

        pitch, yaw = self._head_pose(lm_px, frame.shape)
        gaze_away = abs(yaw) > gaze_yaw_threshold

        # If eye signal is weak (e.g., glasses glare), rely more on non-eye channels.
        glasses_fallback = ear < 0.02 or math.isnan(ear)
        if glasses_fallback:
            ear = calibrated_ear_threshold

        return FrameSignals(
            timestamp=now,
            face_detected=True,
            multiple_faces=multiple_faces,
            ear=ear,
            mar=mar,
            blink_rate=blink_rate,
            blink_detected=blink_detected,
            head_pitch=pitch,
            head_yaw=yaw,
            gaze_away=gaze_away,
            low_light=low_light,
            glasses_fallback=glasses_fallback,
            face_box_area=area,
        )
