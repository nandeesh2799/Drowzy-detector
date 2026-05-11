from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict

from core.feature_extraction import FrameSignals


@dataclass
class FatigueAssessment:
    score_raw: float
    score_smoothed: float
    state: str
    reasons: Dict[str, float]


class FatigueModel:
    def __init__(self, smoothing_window: int = 30) -> None:
        self._history: Deque[float] = deque(maxlen=smoothing_window)

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return max(lo, min(hi, value))

    def assess(self, sig: FrameSignals, baseline: Dict[str, float]) -> FatigueAssessment:
        ear_baseline = baseline.get("ear_baseline", 0.28)
        blink_baseline = baseline.get("blink_rate_baseline", 18.0)

        reasons = {
            "eye_closure": 0.0,
            "yawning": 0.0,
            "blink_anomaly": 0.0,
            "head_pose": 0.0,
            "gaze": 0.0,
            "no_face": 0.0,
        }

        if not sig.face_detected:
            reasons["no_face"] = 45.0
        else:
            ear_drop = max(0.0, (ear_baseline - sig.ear) / max(ear_baseline, 1e-6))
            # Drastically reduced to avoid alerts while looking at camera
            reasons["eye_closure"] = min(5.0, ear_drop * 10.0)

            # Drastically reduced
            reasons["yawning"] = 2.0 if sig.mar > 0.80 else 0.0

            blink_delta = abs(sig.blink_rate - blink_baseline) / max(blink_baseline, 1.0)
            # Drastically reduced
            reasons["blink_anomaly"] = min(2.0, blink_delta * 4.0)

            # Revert head pose to be more sensitive as it counts as distraction/slumping
            if abs(sig.head_pitch) > 18.0:
                reasons["head_pose"] += 15.0
            if abs(sig.head_pitch) > 25.0:
                reasons["head_pose"] += 10.0

            # Revert gaze to be strict (looking left/right)
            reasons["gaze"] = 20.0 if sig.gaze_away else 0.0

            if sig.multiple_faces:
                reasons["gaze"] += 5.0

            if sig.low_light:
                reasons["eye_closure"] += 1.0

        raw = self._clamp(sum(reasons.values()))
        self._history.append(raw)
        smooth = sum(self._history) / len(self._history)

        if smooth < 40.0:
            state = "Alert"
        elif smooth < 70.0:
            state = "Warning"
        else:
            state = "Critical"

        return FatigueAssessment(
            score_raw=raw,
            score_smoothed=smooth,
            state=state,
            reasons=reasons,
        )
