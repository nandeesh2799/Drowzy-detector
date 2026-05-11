from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class DistractionOutput:
    score: float
    label: str
    reason: str


class DistractionDetector:
    def __init__(self, window_seconds: float = 8.0) -> None:
        self.window_seconds = window_seconds
        self._history: Deque[tuple[float, str]] = deque(maxlen=2000)

    def update(
        self,
        timestamp: float,
        yaw: float,
        pitch: float,
        face_detected: bool,
        yaw_threshold: float = 28.0,
        pitch_threshold: float = 24.0,
        warning_threshold: float = 50.0,
        critical_threshold: float = 80.0,
    ) -> DistractionOutput:
        reason = "Focused"
        state = "focused"

        if not face_detected:
            state = "face_out"
            reason = "Face out of frame"
        elif yaw >= yaw_threshold:
            state = "looking_right"
            reason = "Looking right"
        elif yaw <= -yaw_threshold:
            state = "looking_left"
            reason = "Looking left"
        elif pitch > pitch_threshold:
            state = "looking_down"
            reason = "Looking down"

        self._history.append((timestamp, state))
        cutoff = timestamp - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        active = [s for _, s in self._history if s != "focused"]
        ratio = len(active) / max(1, len(self._history))
        score = min(100.0, ratio * 100.0)

        if score >= critical_threshold:
            label = "Critical"
        elif score >= warning_threshold:
            label = "Warning"
        else:
            label = "Alert"

        # Lateral glance events should alert immediately.
        if state in ("looking_left", "looking_right") and label == "Alert":
            label = "Warning"
            score = max(score, warning_threshold)

        if label == "Alert":
            reason = "Focused"
        return DistractionOutput(score=score, label=label, reason=reason)
