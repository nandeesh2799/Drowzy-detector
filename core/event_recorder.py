from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, Tuple
import time

import cv2
import numpy as np


class EventRecorder:
    def __init__(self, output_dir: Path, buffer_seconds: int = 12, fps: int = 20) -> None:
        self.output_dir = output_dir
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self._buffer: Deque[Tuple[float, np.ndarray, Dict[str, float]]] = deque(maxlen=buffer_seconds * fps)
        self._last_event_save = 0.0
        self._cooldown = 25.0

    def add_frame(self, timestamp: float, frame_bgr: np.ndarray, metrics: Dict[str, float]) -> None:
        frame_copy = frame_bgr.copy()
        self._buffer.append((timestamp, frame_copy, metrics))

    def save_event_clip(self, reason: str) -> str | None:
        now = time.time()
        if (now - self._last_event_save) < self._cooldown:
            return None
        if not self._buffer:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        out_file = self.output_dir / f"critical-event-{ts}.mp4"

        sample_frame = self._buffer[0][1]
        h, w = sample_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(out_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self.fps),
            (w, h),
        )

        for _, frame, metrics in self._buffer:
            overlay = frame.copy()
            score = metrics.get("fatigue_score", 0.0)
            distract = metrics.get("distraction_score", 0.0)
            text = f"{reason} | fatigue {score:.1f} | distraction {distract:.1f}"
            cv2.putText(overlay, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            writer.write(overlay)

        writer.release()
        self._last_event_save = now
        return str(out_file)
