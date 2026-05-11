from __future__ import annotations

from types import SimpleNamespace
import time

import cv2
import mediapipe as mp
import numpy as np


class FaceDetectionEngine:
    def __init__(
        self,
        max_num_faces: int = 3,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._face_mesh = None
        self._landmarker = None
        self._use_tasks_api = hasattr(mp, "tasks")
        self._last_video_ts_ms = 0

        if self._use_tasks_api:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            model_path = "face_landmarker.task"
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        else:
            # Legacy path for older mediapipe packages.
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._drawing = mp.solutions.drawing_utils
            self._styles = mp.solutions.drawing_styles

    @staticmethod
    def auto_brightness(frame: np.ndarray, target_mean: float = 120.0) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = float(np.mean(gray))
        if current >= target_mean:
            return frame
        alpha = max(1.0, min(2.0, target_mean / max(current, 1.0)))
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=8)

    def detect(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            now_ms = int(time.time() * 1000)
            if now_ms <= self._last_video_ts_ms:
                now_ms = self._last_video_ts_ms + 1
            self._last_video_ts_ms = now_ms

            res = self._landmarker.detect_for_video(mp_image, now_ms)
            faces = [SimpleNamespace(landmark=lm_list) for lm_list in (res.face_landmarks or [])]
            return SimpleNamespace(multi_face_landmarks=faces)
        return self._face_mesh.process(frame_rgb)

    def draw_face_mesh(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        if face_landmarks is None:
            return frame
        annotated = frame.copy()
        if self._use_tasks_api:
            h, w = annotated.shape[:2]
            points = getattr(face_landmarks, "landmark", face_landmarks)
            for p in points:
                x = int(p.x * w)
                y = int(p.y * h)
                cv2.circle(annotated, (x, y), 1, (0, 255, 255), -1)
        else:
            self._drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=self._mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._styles.get_default_face_mesh_tesselation_style(),
            )
            self._drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=self._mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._styles.get_default_face_mesh_contours_style(),
            )
        return annotated

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
        if self._face_mesh is not None:
            self._face_mesh.close()
