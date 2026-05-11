from __future__ import annotations

from dataclasses import dataclass
import threading
import sys
import time

import pyttsx3


@dataclass
class AlertEvent:
    level: str
    message: str
    timestamp: float


class AlertManager:
    def __init__(self, cooldown_seconds: float = 2.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.distraction_cooldown_seconds = 4.0
        self._last_alert_ts = 0.0
        self._last_distraction_ts = 0.0
        self._tts = pyttsx3.init()
        self._tts.setProperty("rate", 175)

    def _can_alert(self, now: float) -> bool:
        return (now - self._last_alert_ts) >= self.cooldown_seconds

    def trigger(self, fatigue_state: str, score: float) -> AlertEvent | None:
        now = time.time()
        if not self._can_alert(now):
            return None

        if fatigue_state == "Alert":
            return None
        if fatigue_state == "Warning":
            level = "medium" if score >= 55 else "low"
            message = "Attention needed. Please stay focused."
        else:
            level = "high"
            message = "Critical fatigue. Pull over now."

        self._last_alert_ts = now
        self._dispatch(level, message)
        return AlertEvent(level=level, message=message, timestamp=now)

    def trigger_distraction(self, distraction_label: str, reason: str) -> AlertEvent | None:
        now = time.time()
        if distraction_label == "Alert":
            return None
        if (now - self._last_distraction_ts) < self.distraction_cooldown_seconds:
            return None

        if distraction_label == "Warning":
            message = "Distraction detected. Keep eyes on road."
            level = "medium"
        else:
            message = f"Critical distraction. {reason}. Focus on road."
            level = "high"

        self._last_distraction_ts = now
        self._dispatch(level, message)
        return AlertEvent(level=level, message=message, timestamp=now)

    def _dispatch(self, level: str, message: str) -> None:
        try:
            if level in ("low", "medium"):
                # Terminal bell fallback (portable and no external assets needed).
                repeats = 1 if level == "low" else 3
                for _ in range(repeats):
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                    time.sleep(0.03)
            else:
                # Avoid blocking the real-time detection loop.
                threading.Thread(target=self._speak_async, args=(message,), daemon=True).start()
        except Exception:
            # Keep detection alive even if audio/TTS backends fail.
            return

    def _speak_async(self, message: str) -> None:
        try:
            self._tts.say(message)
            self._tts.runAndWait()
        except Exception:
            return
