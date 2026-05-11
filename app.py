from __future__ import annotations

import pathlib
import sys

from streamlit.web import cli as stcli


if __name__ == "__main__":
    ui_app = pathlib.Path(__file__).resolve().parent / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(ui_app), "--server.headless=true"]
    raise SystemExit(stcli.main())
