# Intelligent Driver Monitoring System (MediaPipe + Python)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Production-style, modular driver monitoring system with multi-signal fatigue detection, adaptive calibration, multi-level alerts, and Streamlit analytics dashboard.

## Features

- Multi-signal fatigue pipeline:
  - Eye Aspect Ratio (EAR)
  - Mouth Aspect Ratio (MAR / yawning)
  - Blink-rate trend
  - Head pose (pitch / yaw)
  - Gaze-away detection
- Fatigue score `0-100` with moving-average smoothing
  - `0-40`: Alert
  - `40-70`: Warning
  - `70-100`: Critical
- Smart alerts with cooldown
  - low / medium sound alerts
  - high-level TTS voice warning
- Personal calibration at session start
  - per-user EAR baseline
  - blink baseline
  - adaptive thresholds
- Edge handling
  - no face
  - multi-face (tracks main/largest face)
  - low-light auto brightness correction
  - glasses fallback path
- Session analytics (JSON logs)
  - total session time
  - drowsy duration
  - number of alerts
  - fatigue and blink trends

## Project Structure

```text
.
├── app.py
├── core
│   ├── face_detection.py
│   ├── feature_extraction.py
│   └── fatigue_model.py
├── utils
│   ├── alerts.py
│   ├── analytics.py
│   └── calibration.py
├── ui
│   └── app.py
└── data
    └── session_logs
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the URL printed by Streamlit (usually `http://localhost:8501`).

## Notes

- Existing browser prototype (`index.html` + `app.js`) is preserved.
- Session logs are stored in `data/session_logs/*.json`.
- Baseline profile is stored in `data/baseline_profile.json`.

## License

MIT
# Drowzy-detector
