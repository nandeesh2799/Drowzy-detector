# 🚗 Intelligent Driver Monitoring System (DMS)

[![CI Pipeline](https://github.com/nandeesh2799/Drowzy-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/nandeesh2799/Drowzy-detector/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-12%20Passed-brightgreen?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An enterprise-grade, real-time **Driver Monitoring System (DMS)** engineered with **MediaPipe**, **OpenCV**, and **Streamlit**. Designed for edge automotive compute, fleet management, and vehicle safety, the system provides multi-factor fatigue scoring, off-road distraction detection, predictive time-to-critical forecasting, explainable alerts, and rolling event video recording.

---

## ⚡ Core Capabilities

- **Fatigue & Drowsiness Tracking**: Computes real-time **Eye Aspect Ratio (EAR)**, blink frequency, and micro-sleep duration.
- **Yawn Detection**: Calculates **Mouth Aspect Ratio (MAR)** to identify prolonged yawning episodes.
- **3D Head Pose & Distraction**: Utilizes `cv2.solvePnP` to estimate head pitch and yaw to detect off-road glances, mobile phone usage, and driver slumping.
- **Predictive Risk Forecasting**: Employs an exponential moving average (EMA) linear slope model to project driver fatigue risk 1-5 minutes in advance.
- **Transparent Explainable AI (XAI)**: Generates human-interpretable root causes for alerts (e.g., `CRITICAL: Eye Closure 65% + Yawning 35% (EAR 0.12, MAR 0.85)`).
- **Incident Blackbox Ring-Buffer**: Automatically records 12-second annotated video clips when critical safety thresholds are breached.
- **Dynamic Calibration**: Learns the driver's unique facial baseline (EAR, blink rate, neutral head pose) within 10 seconds of startup.
- **Resilient Model Architecture**: Dynamically detects MediaPipe Tasks API (`face_landmarker.task`) with automatic fallback to bundled face mesh models.

---

## 🏗️ Architecture & Signal Pipeline

```
  ┌─────────────────┐      Auto-Brightness
  │   Video Stream  │ ───────────────────────► [ Brightness Normalizer ]
  └────────┬────────┘                                  │
           │                                           ▼
           │                                 ┌───────────────────┐
           └────────────────────────────────►│ Face & Landmarks  │
                                             └─────────┬─────────┘
                                                       │
         ┌──────────────────────────────┬──────────────┴────────────────┬──────────────────────────────┐
         ▼                              ▼                               ▼                              ▼
  [ EAR / Blinks ]               [ MAR / Yawns ]               [ solvePnP 3D Pose ]            [ Gaze Vector ]
         │                              │                               │                              │
         └──────────────────────────────┼───────────────────────────────┴──────────────────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │    Fatigue / Distraction    │
                         │       Fusion Engine         │
                         └──────────────┬──────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
   [ Predictive Forecast ]    [ Explainability XAI ]     [ Blackbox Event Clip ]
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │   Streamlit Telemetry UI    │
                         └─────────────────────────────┘
```

---

## 🧪 Testing & Verification

The suite includes comprehensive unit tests verifying geometric calculation of EAR, MAR, temporal blink edge detection, fatigue state transitions, distraction thresholds, predictive slope modeling, calibration persistence, and auto-brightness:

```bash
# Run test suite
python3 -m pytest tests/ -v
```

---

## 🚀 Quickstart & Installation

### Option 1: Native Environment

```bash
# Clone repository
git clone https://github.com/nandeesh2799/Drowzy-detector.git
cd Drowzy-detector

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run DMS Application
python3 run.py
```

### Option 2: Docker Containerization

Run the application inside a reproducible Docker container:

```bash
# Build and run with Docker Compose
docker compose up --build

# Open browser at http://localhost:8501
```

---

## 👨‍💻 Author & Engineering Showcase

**Nandeesh N K**  
Full-Stack Developer & AI Systems Engineer — Bengaluru, India  
- 🌐 **Interactive Portfolio:** [NandeeshOS (Live Demo)](https://nandeesh2799-portfolio.vercel.app)
- 💼 **LinkedIn:** [linkedin.com/in/nandeesh-n-k-729773270](https://linkedin.com/in/nandeesh-n-k-729773270)
- 🐙 **GitHub Profile:** [github.com/nandeesh2799](https://github.com/nandeesh2799)
- 📫 **Direct Email:** [nandeeshnk506@gmail.com](mailto:nandeeshnk506@gmail.com)

---

## 📄 License
This project is licensed under the MIT License.
