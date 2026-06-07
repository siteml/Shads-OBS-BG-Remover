@echo off
REM Convenience launcher for Deploy-BGRemoval.ps1.
REM Double-click it (or right-click -> Run as administrator). If it isn't
REM already elevated, it relaunches itself via UAC, then runs the deploy
REM script with the execution policy bypassed and pauses so you can read it.

net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Requesting administrator privileges...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

echo Deploying BG Removal plugin...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Deploy-BGRemoval.ps1"
echo.
pause
