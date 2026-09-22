@echo off
setlocal

echo =======================================================
echo   AMR Intelligence System - Launching Dashboard
echo =======================================================

cd /d "%~dp0"

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python 3.10+ and ensure "Add Python to PATH" is checked.
    pause
    exit /b 1
)

python start.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
