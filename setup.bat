@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo   AMR Intelligence System - Automated Setup (Windows)
echo =======================================================

cd /d "%~dp0"

:: 1. Check for Python installation
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python 3.10+ and ensure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version

:: 2. Create Virtual Environment (.venv)
if not exist ".venv" (
    echo [2/4] Creating virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/4] Virtual environment (.venv) already exists.
)

:: 3. Activate Virtual Environment & Install Requirements
echo [3/4] Installing dependencies from requirements.txt...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: 4. Verify Environment
echo [4/4] Verifying environment & packages...
python -c "import torch, sklearn, streamlit, pandas, shap, plotly, joblib, pytest; print('[SUCCESS] Core packages verified successfully.')"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Environment verification failed.
    pause
    exit /b 1
)

echo =======================================================
echo   Setup Complete! You can now launch the app with:
echo   start.bat
echo =======================================================
pause
