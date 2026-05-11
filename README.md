# Intelligent Driver Monitoring System

AI-powered real-time drowsiness and distraction detection system using MediaPipe and Streamlit. Works across Linux, macOS, and Windows.

## Features
- **Fatigue Detection:** Real-time monitoring of Eye Aspect Ratio (EAR) and Blink Rate.
- **Distraction Detection:** Head pose (Yaw/Pitch) and Gaze-away tracking.
- **Predictive Analytics:** Estimates when fatigue might become critical based on trends.
- **Explainable AI:** Provides reasons for alerts (e.g., "Frequent yawning", "Long eye closure").
- **Session Reports:** Automatic logging of events and session summaries.
- **Cross-Platform:** Optimized camera backends for all major operating systems.

## Prerequisites
- Python 3.10 or higher.
- Webcam (Integrated or External).

## Setup & Installation

### 1. Download MediaPipe Model
This project requires the `face_landmarker.task` model file.
1. Download it from [MediaPipe Google Developers](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task).
2. Place the file in the root directory of this project.

### 2. Virtual Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd drowsy-detector-mediapipe

# Create virtual environment
python -m venv .venv

# Activate environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the App
```bash
python run.py
```
Then open the local URL provided by Streamlit (usually `http://localhost:8501`).

## Platform-Specific Notes

### macOS
- **Permissions:** You may need to grant "Camera" and "Microphone" (for TTS) permissions to your terminal or IDE (e.g., VS Code/Terminal) in `System Settings > Security & Privacy`.
- **Backend:** Uses `AVFOUNDATION` for low-latency capture.

### Windows
- **Backend:** Uses `DirectShow` (DSHOW) for reliable camera initialization.
- **Audio:** Ensure your default audio device is active for voice alerts.

### Linux
- **Backend:** Uses standard V4L2 backend.
- **Dependencies:** May require `libgl1` (e.g., `sudo apt-get install libgl1`).

## Troubleshooting
- **No Camera Detected:** Ensure no other app (Zoom, Teams) is using the camera. Try changing the "Camera" index in the UI.
- **Low FPS:** Reduce browser load or ensure your laptop is plugged into power.
- **Lag:** The app includes optimizations like reduced buffer size and throttled graph updates. If lag persists, try a lower resolution webcam.

## License
MIT
