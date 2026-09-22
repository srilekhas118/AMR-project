"""Create a virtual environment, install dependencies, and prepare data."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
PACKAGES = (
    "torch",
    "sklearn",
    "streamlit",
    "pandas",
    "shap",
    "plotly",
    "joblib",
    "pytest",
)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def log(message: str) -> None:
    print(message, flush=True)


def run(args: list[str], *, env: dict[str, str] | None = None) -> int:
    completed = subprocess.run(args, cwd=ROOT, env=env, check=False)
    return completed.returncode


def run_or_exit(args: list[str], *, env: dict[str, str] | None = None) -> None:
    code = run(args, env=env)
    if code != 0:
        raise SystemExit(code)


def pip_install(py: str) -> None:
    attempts = 3
    for attempt in range(1, attempts + 1):
        code = run([py, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        if code == 0:
            return
        if attempt < attempts:
            log(f"      Install attempt {attempt} failed (Windows file lock is common). Retrying...")
            time.sleep(3)
    log("[ERROR] Failed to install dependencies after retries.")
    raise SystemExit(1)


def main() -> None:
    log("=" * 55)
    log("  AMR Intelligence System - Setup")
    log("=" * 55)

    log("[1/4] Checking Python version...")
    log(f"      {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        log("[ERROR] Python 3.10+ is required.")
        raise SystemExit(1)

    log("[2/4] Creating virtual environment (.venv)...")
    if not venv_python().exists():
        run_or_exit([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        log("      Virtual environment already exists.")

    py = str(venv_python())
    if not REQUIREMENTS.exists():
        log("[ERROR] requirements.txt was not found.")
        raise SystemExit(1)

    log("[3/4] Installing dependencies from requirements.txt...")
    run_or_exit([py, "-m", "pip", "install", "--upgrade", "pip"])
    pip_install(py)

    log("[4/4] Verifying packages and preparing data...")
    imports = ", ".join(PACKAGES)
    run_or_exit([py, "-c", f"import {imports}; print('[SUCCESS] Core packages verified.')"])

    processed = ROOT / "data" / "processed" / "train.csv"
    if not processed.exists():
        log("      Processed data missing — running scripts/setup_data.py...")
        run_or_exit([py, str(ROOT / "scripts" / "setup_data.py")])
    else:
        log("      Processed data already present.")

    log("=" * 55)
    log("  Setup complete. Launch the dashboard with:")
    log("  python start.py")
    log("  or start.bat on Windows")
    log("=" * 55)


if __name__ == "__main__":
    main()
