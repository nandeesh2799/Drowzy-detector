import subprocess
import sys
import os
from pathlib import Path

def run_app():
    """Unified entrypoint for Driver Monitoring System."""
    root = Path(__file__).parent.resolve()
    ui_app = root / "ui" / "app.py"
    
    # Create necessary data directories if they don't exist
    (root / "data" / "events").mkdir(parents=True, exist_ok=True)
    (root / "data" / "session_logs").mkdir(parents=True, exist_ok=True)
    
    print("Checking dependencies...")
    try:
        import streamlit
        import cv2
        import mediapipe
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

    print("Launching Driver Monitoring System...")
    # Use sys.executable to ensure we use the same python interpreter (venv)
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(ui_app), 
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_app()
