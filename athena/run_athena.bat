@echo off
setlocal
cd /d "%~dp0"
echo ========================================
echo   ATHENA 3.0 - Unified CSI Dashboard
echo ========================================
echo.
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Dependency installation failed. Check Python and internet access.
  pause
  exit /b 1
)
echo.
python -m athena
pause
