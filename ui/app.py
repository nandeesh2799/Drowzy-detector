from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path
import time

import cv2
import plotly.express as px
import pandas as pd
import streamlit as st

from core.distraction import DistractionDetector
from core.event_recorder import EventRecorder
from core.explainability import ExplainabilityEngine
from core.face_detection import FaceDetectionEngine
from core.feature_extraction import FeatureExtractor
from core.fatigue_model import FatigueModel
from core.predictive import TemporalPredictor
from utils.alerts import AlertManager
from utils.analytics import SessionAnalytics
from utils.calibration import BaselineProfile, CalibrationManager


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "session_logs"
PROFILE_PATH = ROOT / "data" / "baseline_profile.json"
EVENT_DIR = ROOT / "data" / "events"


BALANCED_SETTINGS = {
    "yaw_offset": 0.0,
    "pitch_threshold": 24.0,
    "warning_threshold": 50.0,
    "critical_threshold": 80.0,
    "fatigue_warning_threshold": 40.0,
    "fatigue_critical_threshold": 70.0,
    "eye_close_factor": 0.95,
    "instant_frames": 4,
    "fatigue_cooldown": 2.0,
    "distraction_cooldown": 4.0,
    "critical_eye_close_seconds": 2.0,
}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: radial-gradient(circle at 10% 10%, #171f42, #0c0f1e 45%, #0a0b13 80%) !important;
            color: #ecf0ff;
        }
        header[data-testid="stHeader"] {
            display: none !important;
            height: 0 !important;
        }
        [data-testid="stToolbar"] {
            display: none !important;
        }
        [data-testid="stDecoration"] {
            display: none !important;
        }
        #MainMenu {
            visibility: hidden !important;
        }
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.1rem !important;
            padding-bottom: 1.2rem !important;
        }
        .control-bar {
            border: 1px solid #2d4374;
            border-radius: 12px;
            background: rgba(10, 16, 32, 0.85);
            padding: 0.35rem 0.5rem;
            margin-bottom: 0.35rem;
        }
        .panel {
            border: 1px solid #263a66;
            border-radius: 12px;
            background: rgba(12, 18, 35, 0.78);
            padding: 0.45rem 0.6rem;
        }
        .metric-big { font-size: clamp(1.2rem, 2.2vw, 2.4rem); font-weight: 800; }
        .green { color: #62ffb1; }
        .yellow { color: #ffd166; }
        .red { color: #ff5f79; }
        .alert-box {
            border: 2px solid #ff4d6d;
            border-radius: 10px;
            background: rgba(255, 77, 109, 0.14);
            padding: 0.45rem 0.55rem;
            margin-bottom: 0.4rem;
            animation: pulse 1s infinite alternate;
        }
        @keyframes pulse { from { opacity: 0.75; } to { opacity: 1.0; } }
        .tiny { font-size: 0.82rem; color: #b9c5e8; }
        .history-item {
            border: 1px solid #2c406e;
            border-radius: 8px;
            padding: 0.25rem 0.4rem;
            margin-bottom: 0.2rem;
            font-size: 0.78rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_state() -> None:
    defaults = {
        "running": False,
        "detector": None,
        "extractor": None,
        "model": None,
        "alerts": None,
        "analytics": None,
        "baseline": None,
        "predictor": None,
        "distraction": None,
        "explainability": None,
        "event_recorder": None,
        "latest_score": 0.0,
        "latest_distraction": 0.0,
        "last_alert_msg": "None",
        "alert_history": [],
        "last_explanation": "No active alerts",
        "predicted_text": "Collecting data...",
        "chart_scores": deque(maxlen=240),
        "chart_blinks": deque(maxlen=240),
        "last_chart_refresh_ts": 0.0,
        "eye_closed_streak": 0,
        "head_tilt_streak": 0,
        "eye_closed_start_ts": None,
        "camera_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_data(ttl=30)
def get_available_cameras():
    # Only check first 4 ports to keep it snappy
    working_ports = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                working_ports.append(i)
            cap.release()
    return working_ports


def ensure_alert_manager_compat() -> None:
    alerts = st.session_state.get("alerts")
    if alerts is None:
        return
    if not hasattr(alerts, "trigger_distraction"):
        # Streamlit can keep old class instances across hot reloads.
        st.session_state.alerts = AlertManager(cooldown_seconds=2.0)


def calibrate_user(
    cap: cv2.VideoCapture,
    detector: FaceDetectionEngine,
    extractor: FeatureExtractor,
    calibration_seconds: int = 5,
) -> BaselineProfile:
    calib = CalibrationManager(PROFILE_PATH)
    start = time.time()
    while time.time() - start < calibration_seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = detector.auto_brightness(frame)
        res = detector.detect(frame)
        sig = extractor.compute_signals(
            frame=frame,
            multi_face_landmarks=res.multi_face_landmarks,
            calibrated_ear_threshold=0.22,
            gaze_yaw_threshold=24.0,
        )
        if sig.face_detected:
            calib.add_sample(
                ear=sig.ear,
                blink_rate=sig.blink_rate,
                head_pitch=sig.head_pitch,
                head_yaw=sig.head_yaw,
            )
    profile = calib.build_profile()
    calib.save_profile(profile)
    return profile


def start_session() -> None:
    detector = FaceDetectionEngine()
    extractor = FeatureExtractor(blink_window_seconds=60.0)
    model = FatigueModel(smoothing_window=35)
    alerts = AlertManager(cooldown_seconds=2.0)
    analytics = SessionAnalytics(DATA_DIR)
    predictor = TemporalPredictor(window_seconds=60, temporal_weight=0.28)
    distraction = DistractionDetector(window_seconds=8.0)
    explainability = ExplainabilityEngine()
    event_recorder = EventRecorder(EVENT_DIR, buffer_seconds=12, fps=20)
    st.session_state.detector = detector
    st.session_state.extractor = extractor
    st.session_state.model = model
    st.session_state.alerts = alerts
    st.session_state.analytics = analytics
    st.session_state.predictor = predictor
    st.session_state.distraction = distraction
    st.session_state.explainability = explainability
    st.session_state.event_recorder = event_recorder
    st.session_state.alert_history = []
    st.session_state.chart_scores = deque(maxlen=240)
    st.session_state.chart_blinks = deque(maxlen=240)
    st.session_state.running = True


def stop_session() -> None:
    st.session_state.running = False
    if st.session_state.detector:
        st.session_state.detector.close()
    if st.session_state.analytics:
        summary = st.session_state.analytics.finalize()
        st.success(
            f"Session saved. Time: {summary.total_session_time_sec:.1f}s | "
            f"Drowsy: {summary.drowsy_duration_sec:.1f}s | Alerts: {summary.total_alerts}"
        )


def run_live_monitor() -> None:
    ensure_alert_manager_compat()
    st.markdown("#### Live Monitoring")
    top_metrics = st.container()
    live_row = st.container()
    explain_row = st.container()
    charts_row = st.container()

    with top_metrics:
        m1, m2, m3, m4 = st.columns(4, gap="small")
        score_metric = m1.empty()
        distraction_metric = m2.empty()
        prediction_metric = m3.empty()
        fps_metric = m4.empty()

    with live_row:
        c1, c2 = st.columns([2, 1], gap="small")
        frame_slot = c1.empty()
        right_slot = c2.empty()

    with explain_row:
        left_slot = st.empty()

    with charts_row:
        b1, b2 = st.columns(2, gap="small")
        trend_slot = b1.empty()
        blink_slot = b2.empty()

    cap = cv2.VideoCapture(st.session_state.camera_index)
    if not cap.isOpened():
        st.error(f"Unable to access webcam at index {st.session_state.camera_index}.")
        st.session_state.running = False
        return
    
    # Optimization: Use lower resolution for faster processing if needed, 
    # but 640x480 is usually fine. Let's ensure it's not trying something higher.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Optimization: Reduce buffer size to minimize lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    st.info("Calibration (5s): keep head straight and eyes open.")
    baseline = calibrate_user(cap, st.session_state.detector, st.session_state.extractor, 5)
    st.session_state.baseline = baseline
    st.success("Monitoring started.")

    prev_ts = time.time()
    frame_count = 0
    while st.session_state.running:
        ok, frame = cap.read()
        if not ok:
            continue
        
        frame_count += 1
        # Skip every other frame for processing to boost FPS if needed
        # if frame_count % 2 != 0: continue

        frame = st.session_state.detector.auto_brightness(frame)
        res = st.session_state.detector.detect(frame)

        sig = st.session_state.extractor.compute_signals(
            frame=frame,
            multi_face_landmarks=res.multi_face_landmarks,
            calibrated_ear_threshold=baseline.ear_threshold,
            gaze_yaw_threshold=baseline.gaze_yaw_threshold,
        )

        normalized_yaw = sig.head_yaw - baseline.head_yaw_baseline
        normalized_pitch = sig.head_pitch - baseline.head_pitch_baseline
        yaw_threshold = max(12.0, baseline.gaze_yaw_threshold + BALANCED_SETTINGS["yaw_offset"])
        sig_norm = replace(
            sig,
            head_yaw=normalized_yaw,
            head_pitch=normalized_pitch,
            gaze_away=abs(normalized_yaw) > yaw_threshold,
        )

        # Dynamic cooldowns and strictness tuning by mode.
        st.session_state.alerts.cooldown_seconds = BALANCED_SETTINGS["fatigue_cooldown"]
        st.session_state.alerts.distraction_cooldown_seconds = BALANCED_SETTINGS["distraction_cooldown"]

        assessment = st.session_state.model.assess(sig_norm, baseline.as_dict())
        predictive = st.session_state.predictor.update(
            timestamp=sig_norm.timestamp,
            fatigue_score=assessment.score_smoothed,
            features={
                "ear": sig_norm.ear,
                "mar": sig_norm.mar,
                "blink_rate": sig_norm.blink_rate,
                "yaw": sig_norm.head_yaw,
                "pitch": sig_norm.head_pitch,
            },
        )
        distraction = st.session_state.distraction.update(
            timestamp=sig_norm.timestamp,
            yaw=sig_norm.head_yaw,
            pitch=sig_norm.head_pitch,
            face_detected=sig_norm.face_detected,
            yaw_threshold=yaw_threshold,
            pitch_threshold=BALANCED_SETTINGS["pitch_threshold"],
            warning_threshold=BALANCED_SETTINGS["warning_threshold"],
            critical_threshold=BALANCED_SETTINGS["critical_threshold"],
        )

        fused_score = (0.75 * assessment.score_smoothed) + (0.25 * predictive.temporal_score)
        fused_score = max(0.0, min(100.0, fused_score))

        eye_close_threshold = baseline.ear_threshold * BALANCED_SETTINGS["eye_close_factor"]
        if sig_norm.ear <= eye_close_threshold:
            st.session_state.eye_closed_streak += 1
            if st.session_state.eye_closed_start_ts is None:
                st.session_state.eye_closed_start_ts = sig_norm.timestamp
        else:
            st.session_state.eye_closed_streak = 0
            st.session_state.eye_closed_start_ts = None

        if abs(sig_norm.head_pitch) >= BALANCED_SETTINGS["pitch_threshold"]:
            st.session_state.head_tilt_streak += 1
        else:
            st.session_state.head_tilt_streak = 0

        if st.session_state.eye_closed_streak >= BALANCED_SETTINGS["instant_frames"]:
            fused_score = max(fused_score, 72.0)
        elif st.session_state.head_tilt_streak >= BALANCED_SETTINGS["instant_frames"]:
            fused_score = max(fused_score, 68.0)

        fatigue_warn = BALANCED_SETTINGS["fatigue_warning_threshold"]
        fatigue_critical = BALANCED_SETTINGS["fatigue_critical_threshold"]
        if fused_score < fatigue_warn:
            fused_state = "Alert"
        elif fused_score < fatigue_critical:
            fused_state = "Warning"
        else:
            fused_state = "Critical"

        eye_closed_duration = 0.0
        if st.session_state.eye_closed_start_ts is not None:
            eye_closed_duration = sig_norm.timestamp - st.session_state.eye_closed_start_ts

        can_emit_critical_fatigue = (
            fused_state == "Critical"
            and eye_closed_duration >= BALANCED_SETTINGS["critical_eye_close_seconds"]
        )
        fatigue_alert_state = "Warning" if fused_state == "Critical" and not can_emit_critical_fatigue else fused_state

        explain = st.session_state.explainability.explain(
            fatigue_state=fused_state,
            reasons=assessment.reasons,
            blink_rate=sig_norm.blink_rate,
            ear=sig_norm.ear,
            mar=sig_norm.mar,
            distraction_reason=distraction.reason,
        )
        st.session_state.last_explanation = explain.summary

        event = st.session_state.alerts.trigger(fatigue_alert_state, fused_score)
        if event:
            st.session_state.analytics.log_alert(event.level, event.message, event.timestamp)
            st.session_state.last_alert_msg = f"{event.level.upper()}: {event.message}"
            st.session_state.alert_history = [st.session_state.last_alert_msg, *st.session_state.alert_history[:3]]

        distraction_event = st.session_state.alerts.trigger_distraction(
            distraction_label=distraction.label,
            reason=distraction.reason,
        )
        if distraction_event:
            msg = f"{distraction_event.level.upper()}: {distraction_event.message}"
            st.session_state.analytics.log_alert(distraction_event.level, distraction_event.message, distraction_event.timestamp)
            st.session_state.last_alert_msg = msg
            st.session_state.alert_history = [msg, *st.session_state.alert_history[:3]]

        if can_emit_critical_fatigue:
            clip_path = st.session_state.event_recorder.save_event_clip(reason="Critical fatigue")
            if clip_path:
                msg = f"CLIP SAVED: {Path(clip_path).name}"
                st.session_state.alert_history = [msg, *st.session_state.alert_history[:3]]

        main_face = None
        if res.multi_face_landmarks:
            main_face, _, _ = st.session_state.extractor.select_main_face(
                res.multi_face_landmarks, frame.shape[1], frame.shape[0]
            )
        annotated = st.session_state.detector.draw_face_mesh(frame, main_face)

        now = time.time()
        fps = 1.0 / max(1e-6, now - prev_ts)
        prev_ts = now

        if predictive.fatigue_in_minutes is None:
            st.session_state.predicted_text = "Fatigue trend stable"
        else:
            st.session_state.predicted_text = f"Fatigue likely in {predictive.fatigue_in_minutes:.1f} min"

        bgr = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_slot.image(bgr, channels="RGB", use_container_width=True, caption="Live webcam + face mesh")

        fatigue_color = "green" if fused_state == "Alert" else ("yellow" if fused_state == "Warning" else "red")
        distraction_color = "green" if distraction.label == "Alert" else ("yellow" if distraction.label == "Warning" else "red")

        score_metric.metric("Fatigue State", f"{fused_state} ({fused_score:.1f})")
        distraction_metric.metric("Distraction", f"{distraction.label} ({distraction.score:.1f})")
        prediction_metric.metric("Prediction", st.session_state.predicted_text)
        fps_metric.metric("FPS", f"{fps:.1f}")

        left_slot.markdown(
            f"""
            <div class="panel">
              <div class="tiny">Signal Breakdown</div>
              <div style="font-size:0.95rem;">
                <span class="{fatigue_color}"><b>Fatigue:</b> {fused_state}</span> |
                <span class="{distraction_color}"><b>Distraction:</b> {distraction.reason}</span>
              </div>
              <div class="tiny" style="margin-top:0.5rem;">
                EAR {sig_norm.ear:.2f} | MAR {sig_norm.mar:.2f} | Blink {sig_norm.blink_rate:.1f}/min |
                Yaw {sig_norm.head_yaw:.1f} | Pitch {sig_norm.head_pitch:.1f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        history = st.session_state.alert_history or ["No alerts yet"]
        history_html = "".join(f"<div class='history-item'>{h}</div>" for h in history[:4])
        right_slot.markdown(
            f"""
            <div class="alert-box"><b>ALERT PANEL</b><br>{st.session_state.last_alert_msg}</div>
            <div class="panel">
              <div class="tiny">Recent Alerts</div>
              {history_html}
              <div class="tiny" style="margin-top:0.35rem;">Explainable AI</div>
              <div style="font-size:0.84rem;">{st.session_state.last_explanation}</div>
              <div class="tiny" style="margin-top:0.4rem;">
                Gaze: {('Away' if sig_norm.gaze_away else 'On-road')} | Distraction: {distraction.reason}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.chart_scores.append({"t": sig_norm.timestamp, "fatigue": fused_score})
        st.session_state.chart_blinks.append({"t": sig_norm.timestamp, "blink": sig_norm.blink_rate})
        score_df = pd.DataFrame(st.session_state.chart_scores)
        blink_df = pd.DataFrame(st.session_state.chart_blinks)
        # Plotly rerenders are expensive; refresh charts at lower frequency to reduce lag.
        if (now - st.session_state.last_chart_refresh_ts) >= 1.5:
            st.session_state.last_chart_refresh_ts = now
            if len(score_df) > 2:
                score_df["x"] = score_df["t"] - score_df["t"].iloc[0]
                fig1 = px.line(score_df, x="x", y="fatigue", template="plotly_dark")
                fig1.update_layout(height=240, margin=dict(l=5, r=5, t=30, b=20), showlegend=False, title="Fatigue Trend")
                trend_slot.plotly_chart(fig1, use_container_width=True)
            if len(blink_df) > 2:
                blink_df["x"] = blink_df["t"] - blink_df["t"].iloc[0]
                fig2 = px.line(blink_df, x="x", y="blink", template="plotly_dark")
                fig2.update_layout(height=240, margin=dict(l=5, r=5, t=30, b=20), showlegend=False, title="Blink Trend")
                blink_slot.plotly_chart(fig2, use_container_width=True)

        st.session_state.event_recorder.add_frame(
            timestamp=sig.timestamp,
            frame_bgr=annotated,
            metrics={"fatigue_score": fused_score, "distraction_score": distraction.score},
        )

        st.session_state.analytics.log_frame(
            {
                "timestamp": sig_norm.timestamp,
                "score": fused_score,
                "score_raw": assessment.score_raw,
                "state": fused_state,
                "ear": sig_norm.ear,
                "mar": sig_norm.mar,
                "blink_rate": sig_norm.blink_rate,
                "yaw": sig_norm.head_yaw,
                "pitch": sig_norm.head_pitch,
                "gaze_away": sig_norm.gaze_away,
                "face_detected": sig_norm.face_detected,
                "low_light": sig_norm.low_light,
                "distraction_score": distraction.score,
                "prediction_minutes": predictive.fatigue_in_minutes or -1,
                "temporal_score": predictive.temporal_score,
            }
        )

        time.sleep(0.001)

    cap.release()


def main() -> None:
    st.set_page_config(page_title="Driver Monitoring AI", page_icon=":car:", layout="wide", initial_sidebar_state="collapsed")
    apply_theme()
    ensure_state()
    ensure_alert_manager_compat()

    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns([2, 1, 1, 1], gap="small")
    with h1:
        st.markdown("## Intelligent Driver Monitoring System")
    with h2:
        cameras = get_available_cameras()
        if not cameras:
            st.error("No cameras detected")
            default_idx = 0
        else:
            default_idx = cameras.index(st.session_state.camera_index) if st.session_state.camera_index in cameras else 0
        
        selected_cam = st.selectbox(
            "Camera",
            options=cameras if cameras else [0],
            index=default_idx,
            key="camera_select_box",
            disabled=st.session_state.running
        )
        st.session_state.camera_index = selected_cam
    with h3:
        if st.button("Start Session", use_container_width=True, disabled=st.session_state.running):
            start_session()
    with h4:
        if st.button("Stop Session", use_container_width=True, disabled=not st.session_state.running):
            stop_session()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.running:
        run_live_monitor()
    else:
        st.info("Single-screen dashboard ready. Click Start Session.")


if __name__ == "__main__":
    main()
