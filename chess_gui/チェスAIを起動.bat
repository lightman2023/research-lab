@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Python environment was not found.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "app.py"

