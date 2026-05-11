from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple
import numpy as np


@dataclass
class PredictiveOutput:
    fatigue_in_minutes: float | None
    trend_slope_per_min: float
    risk_label: str
    temporal_score: float


class TemporalPredictor:
    def __init__(self, window_seconds: int = 60, temporal_weight: float = 0.35) -> None:
        self.window_seconds = window_seconds
        self.temporal_weight = temporal_weight
        self._buffer: Deque[Dict[str, float]] = deque(maxlen=1200)
        self._ema_score = 0.0

    def update(self, timestamp: float, fatigue_score: float, features: Dict[str, float]) -> PredictiveOutput:
        row = {"timestamp": timestamp, "score": fatigue_score, **features}
        self._buffer.append(row)

        cutoff = timestamp - float(self.window_seconds)
        while self._buffer and self._buffer[0]["timestamp"] < cutoff:
            self._buffer.popleft()

        alpha = self.temporal_weight
        self._ema_score = (alpha * fatigue_score) + ((1.0 - alpha) * self._ema_score)

        minutes, slope = self._predict_time_to_threshold(target=70.0)
        risk = "Stable"
        if minutes is not None and minutes <= 2.0:
            risk = "High risk"
        elif minutes is not None and minutes <= 5.0:
            risk = "Rising risk"

        return PredictiveOutput(
            fatigue_in_minutes=minutes,
            trend_slope_per_min=slope * 60.0,
            risk_label=risk,
            temporal_score=float(self._ema_score),
        )

    def _predict_time_to_threshold(self, target: float) -> Tuple[float | None, float]:
        if len(self._buffer) < 20:
            return None, 0.0

        t0 = self._buffer[0]["timestamp"]
        x = np.array([r["timestamp"] - t0 for r in self._buffer], dtype=np.float64)
        y = np.array([r["score"] for r in self._buffer], dtype=np.float64)
        if np.ptp(x) < 1e-6:
            return None, 0.0

        slope, intercept = np.polyfit(x, y, 1)
        current = float(y[-1])
        if slope <= 0.0:
            return None, float(slope)

        seconds_to_target = (target - current) / slope
        if seconds_to_target < 0:
            return 0.0, float(slope)
        return max(0.0, seconds_to_target / 60.0), float(slope)
