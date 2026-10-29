import os
import json
import math
import tempfile
from pathlib import Path
import pytest
import numpy as np

from core.feature_extraction import (
    FeatureExtractor,
    FrameSignals,
    LEFT_EYE,
    RIGHT_EYE,
    MOUTH,
)
from core.fatigue_model import FatigueModel, FatigueAssessment
from core.distraction import DistractionDetector, DistractionOutput
from core.predictive import TemporalPredictor, PredictiveOutput
from core.explainability import ExplainabilityEngine
from core.event_recorder import EventRecorder
from utils.calibration import CalibrationManager, BaselineProfile
from utils.analytics import SessionAnalytics
from core.face_detection import FaceDetectionEngine


class TestFeatureExtraction:
    def test_ear_calculation_open_and_closed(self):
        extractor = FeatureExtractor()

        # Construct synthetic open eye coordinates
        # Indices: [33, 160, 158, 133, 153, 144]
        # p0=(0,0), p3=(10,0) -> width = 10
        # p1=(3,3), p5=(3,-3) -> height1 = 6
        # p2=(7,3), p4=(7,-3) -> height2 = 6
        # EAR = (6 + 6) / (2 * 10) = 0.60
        open_eye_lm = {
            33: (0.0, 0.0),
            160: (3.0, 3.0),
            158: (7.0, 3.0),
            133: (10.0, 0.0),
            153: (7.0, -3.0),
            144: (3.0, -3.0),
        }
        open_ear = extractor._ear(open_eye_lm, LEFT_EYE)
        assert pytest.approx(open_ear, rel=1e-3) == 0.60

        # Construct synthetic closed eye coordinates
        # p1=(3, 0.2), p5=(3, -0.2) -> height1 = 0.4
        # p2=(7, 0.2), p4=(7, -0.2) -> height2 = 0.4
        # EAR = (0.4 + 0.4) / (2 * 10) = 0.04
        closed_eye_lm = {
            33: (0.0, 0.0),
            160: (3.0, 0.2),
            158: (7.0, 0.2),
            133: (10.0, 0.0),
            153: (7.0, -0.2),
            144: (3.0, -0.2),
        }
        closed_ear = extractor._ear(closed_eye_lm, LEFT_EYE)
        assert pytest.approx(closed_ear, rel=1e-3) == 0.04
        assert closed_ear < 0.15

    def test_mar_calculation_mouth_opening(self):
        extractor = FeatureExtractor()

        # Indices in MOUTH: [61, 81, 13, 311, 291, 178, 14, 402]
        # horizontal: p0 (61) to p4 (291)
        # vertical pairs: (p1, p7), (p2, p6), (p3, p5)
        # Normal mouth: width = 20, vertical = 2
        normal_mouth = {
            61: (-10.0, 0.0),
            81: (-5.0, 1.0),
            13: (0.0, 1.0),
            311: (5.0, 1.0),
            291: (10.0, 0.0),
            178: (5.0, -1.0),
            14: (0.0, -1.0),
            402: (-5.0, -1.0),
        }
        normal_mar = extractor._mar(normal_mouth)
        assert normal_mar < 0.25

        # Yawn mouth: width = 20, vertical gap = 16
        yawn_mouth = {
            61: (-10.0, 0.0),
            81: (-5.0, 8.0),
            13: (0.0, 8.0),
            311: (5.0, 8.0),
            291: (10.0, 0.0),
            178: (5.0, -8.0),
            14: (0.0, -8.0),
            402: (-5.0, -8.0),
        }
        yawn_mar = extractor._mar(yawn_mouth)
        assert yawn_mar > 0.70

    def test_blink_rate_tracking(self):
        extractor = FeatureExtractor(blink_window_seconds=60.0)

        # Initial state: eye open
        rate, detected = extractor._blink_rate(now=10.0, eye_closed=False)
        assert rate == 0.0
        assert not detected

        # Blink 1: rising edge
        rate, detected = extractor._blink_rate(now=10.1, eye_closed=True)
        assert detected is True
        assert rate == 1.0

        # Holding closed: no new blink edge
        rate, detected = extractor._blink_rate(now=10.2, eye_closed=True)
        assert detected is False
        assert rate == 1.0

        # Eye opens
        rate, detected = extractor._blink_rate(now=10.3, eye_closed=False)
        assert detected is False

        # Blink 2: rising edge
        rate, detected = extractor._blink_rate(now=12.0, eye_closed=True)
        assert detected is True
        assert rate == 2.0


class TestFatigueModel:
    def test_alert_state_with_normal_signals(self):
        model = FatigueModel(smoothing_window=5)
        normal_sig = FrameSignals(
            timestamp=100.0,
            face_detected=True,
            multiple_faces=False,
            ear=0.29,
            mar=0.15,
            blink_rate=18.0,
            blink_detected=False,
            head_pitch=0.0,
            head_yaw=0.0,
            gaze_away=False,
            low_light=False,
            glasses_fallback=False,
            face_box_area=50000.0,
        )
        baseline = {"ear_baseline": 0.28, "blink_rate_baseline": 18.0}

        assessment = model.assess(normal_sig, baseline)
        assert assessment.state == "Alert"
        assert assessment.score_raw < 40.0
        assert assessment.score_smoothed < 40.0

    def test_critical_state_on_missing_face_or_severe_fatigue(self):
        model = FatigueModel(smoothing_window=3)
        no_face_sig = FrameSignals(
            timestamp=100.0,
            face_detected=False,
            multiple_faces=False,
            ear=0.0,
            mar=0.0,
            blink_rate=0.0,
            blink_detected=False,
            head_pitch=0.0,
            head_yaw=0.0,
            gaze_away=False,
            low_light=False,
            glasses_fallback=False,
            face_box_area=0.0,
        )
        baseline = {"ear_baseline": 0.28, "blink_rate_baseline": 18.0}

        assessment = model.assess(no_face_sig, baseline)
        assert assessment.reasons["no_face"] == 45.0
        assert assessment.score_raw >= 45.0


class TestDistractionDetector:
    def test_head_turn_and_gaze_distraction(self):
        detector = DistractionDetector(window_seconds=10.0)

        # 5 seconds focused
        for t in range(5):
            res = detector.update(timestamp=float(t), yaw=0.0, pitch=0.0, face_detected=True)
            assert res.label == "Alert"

        # Next 10 seconds driver looks right beyond threshold (e.g. looking at phone)
        for t in range(5, 15):
            res = detector.update(timestamp=float(t), yaw=35.0, pitch=5.0, face_detected=True)

        assert res.score > 50.0
        assert res.label in ("Warning", "Critical")
        assert res.reason == "Looking right"


class TestTemporalPredictor:
    def test_slope_and_risk_projection(self):
        predictor = TemporalPredictor(window_seconds=60)

        # Simulate steadily rising fatigue from 10 to 65
        for i in range(25):
            t = float(i * 2)  # every 2 seconds
            score = 10.0 + (i * 2.2)
            out = predictor.update(timestamp=t, fatigue_score=score, features={"ear": 0.20})

        assert out.trend_slope_per_min > 0.0
        assert out.temporal_score > 0.0
        assert out.risk_label in ("Stable", "Rising risk", "High risk")


class TestCalibrationAndAnalytics:
    def test_calibration_manager_build_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.json"
            mgr = CalibrationManager(profile_path)

            # Add calibration samples
            for ear in [0.28, 0.30, 0.32, 0.30]:
                mgr.add_sample(ear=ear, blink_rate=16.0, head_pitch=2.0, head_yaw=-1.0)

            profile = mgr.build_profile()
            assert pytest.approx(profile.ear_baseline, rel=1e-2) == 0.30
            assert profile.ear_threshold == pytest.approx(0.30 * 0.75, rel=1e-2)

            mgr.save_profile(profile)
            assert profile_path.exists()

            loaded = mgr.load_profile()
            assert pytest.approx(loaded.ear_baseline, rel=1e-2) == 0.30
            assert pytest.approx(loaded.blink_rate_baseline, rel=1e-2) == 16.0

    def test_session_analytics_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analytics = SessionAnalytics(Path(tmpdir))

            analytics.log_frame({"timestamp": 100.0, "state": "Alert", "score": 10.0})
            analytics.log_frame({"timestamp": 102.0, "state": "Warning", "score": 55.0})
            analytics.log_frame({"timestamp": 105.0, "state": "Critical", "score": 85.0})
            analytics.log_alert("Warning", "High fatigue detected", timestamp=102.5)

            summary = analytics.finalize()
            assert summary.total_alerts == 1
            assert summary.drowsy_duration_sec == pytest.approx(5.0)  # (102-100) + (105-102) = 5.0

            saved_file = Path(tmpdir) / f"{summary.session_id}.json"
            assert saved_file.exists()


class TestAutoBrightness:
    def test_auto_brightness_scaling(self):
        dark_frame = np.ones((100, 100, 3), dtype=np.uint8) * 40
        brightened = FaceDetectionEngine.auto_brightness(dark_frame, target_mean=120.0)

        assert np.mean(brightened) > np.mean(dark_frame)


class TestExplainabilityAndEvents:
    def test_explainability_engine_formatting(self):
        engine = ExplainabilityEngine()
        reasons = {"eye_closure": 25.0, "yawning": 15.0, "head_pose": 10.0}

        out = engine.explain(
            fatigue_state="Critical",
            reasons=reasons,
            blink_rate=22.0,
            ear=0.12,
            mar=0.85,
            distraction_reason="Looking left",
        )

        assert "CRITICAL" in out.summary
        assert "Eye Closure" in out.top_reasons[0]
        assert "EAR 0.12" in out.summary
        assert "Looking left" in out.summary

    def test_event_recorder_cooldown_and_empty_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = EventRecorder(Path(tmpdir), buffer_seconds=2, fps=10)

            # Empty buffer should return None
            assert recorder.save_event_clip("Test event") is None

            # Add frame
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            recorder.add_frame(100.0, dummy_frame, {"fatigue_score": 75.0, "distraction_score": 20.0})

            # First clip saved
            clip = recorder.save_event_clip("Critical fatigue")
            assert clip is not None
            assert os.path.exists(clip)

            # Second immediate save should be throttled by cooldown
            assert recorder.save_event_clip("Duplicate trigger") is None

