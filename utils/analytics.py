from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import time
import uuid


@dataclass
class SessionSummary:
    session_id: str
    started_at: float
    ended_at: float
    total_session_time_sec: float
    drowsy_duration_sec: float
    total_alerts: int


class SessionAnalytics:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.session_id = str(uuid.uuid4())
        self.started_at = time.time()
        self.rows: List[Dict] = []
        self.alerts: List[Dict] = []
        self.drowsy_seconds = 0.0
        self._last_timestamp = self.started_at

    def log_frame(self, payload: Dict) -> None:
        now = float(payload.get("timestamp", time.time()))
        state = payload.get("state", "Alert")
        delta = max(0.0, now - self._last_timestamp)
        if state in ("Warning", "Critical"):
            self.drowsy_seconds += delta
        self._last_timestamp = now
        self.rows.append(payload)

    def log_alert(self, level: str, message: str, timestamp: float) -> None:
        self.alerts.append({"level": level, "message": message, "timestamp": timestamp})

    def finalize(self) -> SessionSummary:
        ended = time.time()
        total = ended - self.started_at
        summary = SessionSummary(
            session_id=self.session_id,
            started_at=self.started_at,
            ended_at=ended,
            total_session_time_sec=total,
            drowsy_duration_sec=self.drowsy_seconds,
            total_alerts=len(self.alerts),
        )
        self._persist(summary)
        return summary

    def _persist(self, summary: SessionSummary) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "summary": summary.__dict__,
            "alerts": self.alerts,
            "timeseries": self.rows,
        }
        out_path = self.data_dir / f"{summary.session_id}.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
