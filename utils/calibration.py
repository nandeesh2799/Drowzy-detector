from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import statistics


@dataclass
class BaselineProfile:
    ear_baseline: float
    ear_threshold: float
    blink_rate_baseline: float
    gaze_yaw_threshold: float
    head_pitch_baseline: float
    head_yaw_baseline: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "ear_baseline": self.ear_baseline,
            "ear_threshold": self.ear_threshold,
            "blink_rate_baseline": self.blink_rate_baseline,
            "gaze_yaw_threshold": self.gaze_yaw_threshold,
            "head_pitch_baseline": self.head_pitch_baseline,
            "head_yaw_baseline": self.head_yaw_baseline,
        }


class CalibrationManager:
    def __init__(self, profile_path: Path) -> None:
        self.profile_path = profile_path
        self._ear_values: List[float] = []
        self._blink_values: List[float] = []
        self._pitch_values: List[float] = []
        self._yaw_values: List[float] = []

    def add_sample(self, ear: float, blink_rate: float, head_pitch: float = 0.0, head_yaw: float = 0.0) -> None:
        if ear > 0:
            self._ear_values.append(ear)
        if blink_rate >= 0:
            self._blink_values.append(blink_rate)
        self._pitch_values.append(float(head_pitch))
        self._yaw_values.append(float(head_yaw))

    def build_profile(self) -> BaselineProfile:
        ear_baseline = statistics.fmean(self._ear_values) if self._ear_values else 0.28
        blink_baseline = statistics.fmean(self._blink_values) if self._blink_values else 18.0
        ear_threshold = max(0.15, ear_baseline * 0.75)
        # Less strict to reduce false off-road alerts in real use.
        gaze_yaw_threshold = 24.0
        head_pitch_baseline = statistics.fmean(self._pitch_values) if self._pitch_values else 0.0
        head_yaw_baseline = statistics.fmean(self._yaw_values) if self._yaw_values else 0.0
        return BaselineProfile(
            ear_baseline=ear_baseline,
            ear_threshold=ear_threshold,
            blink_rate_baseline=blink_baseline,
            gaze_yaw_threshold=gaze_yaw_threshold,
            head_pitch_baseline=head_pitch_baseline,
            head_yaw_baseline=head_yaw_baseline,
        )

    def save_profile(self, profile: BaselineProfile) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(json.dumps(profile.as_dict(), indent=2), encoding="utf-8")

    def load_profile(self) -> BaselineProfile:
        if not self.profile_path.exists():
            return self.build_profile()
        payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
        return BaselineProfile(
            ear_baseline=float(payload.get("ear_baseline", 0.28)),
            ear_threshold=float(payload.get("ear_threshold", 0.22)),
            blink_rate_baseline=float(payload.get("blink_rate_baseline", 18.0)),
            gaze_yaw_threshold=float(payload.get("gaze_yaw_threshold", 24.0)),
            head_pitch_baseline=float(payload.get("head_pitch_baseline", 0.0)),
            head_yaw_baseline=float(payload.get("head_yaw_baseline", 0.0)),
        )
