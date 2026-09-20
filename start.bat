@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo   AMR Intelligence System - Launching Dashboard
echo =======================================================

cd /d "%~dp0"

:: Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (.venv) not found.
    echo Please run setup.bat first to initialize the environment and install dependencies.
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

echo Starting Streamlit application on http://localhost:8501 ...
echo (Press Ctrl+C in this terminal to stop the server)
echo.

python -m streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with an error code: %ERRORLEVEL%
    pause
)
