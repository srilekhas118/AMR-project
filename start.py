"""Launch the Streamlit AMR Intelligence dashboard."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
APP = ROOT / "app.py"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main() -> None:
    print("=" * 55)
    print("  AMR Intelligence System - Launching Dashboard")
    print("=" * 55)

    py = venv_python()
    if not py.exists():
        print("[ERROR] Virtual environment (.venv) not found.")
        print("Run python setup.py first (or setup.bat on Windows).")
        raise SystemExit(1)

    if not APP.exists():
        print("[ERROR] app.py was not found.")
        raise SystemExit(1)

    print("Starting Streamlit on http://localhost:8501 ...", flush=True)
    print("(Press Ctrl+C in this terminal to stop the server)", flush=True)
    print(flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    cmd = [str(py), "-m", "streamlit", "run", str(APP), "--server.port", "8501"]
    if env.get("STREAMLIT_SERVER_HEADLESS", "").lower() in {"1", "true", "yes"}:
        cmd.extend(["--server.headless", "true"])
    completed = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        print(f"[ERROR] Application exited with code {completed.returncode}.")
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
