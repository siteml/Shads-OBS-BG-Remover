@echo off
REM Fetches the OpenVINO 2026.1 Windows runtime into deps\openvino
REM (run from anywhere - it always targets the deps folder beside this script).
setlocal
cd /d "%~dp0"
if not exist deps mkdir deps
cd deps

if exist openvino (
  echo An "openvino" folder already exists in deps\.
  echo Delete or rename it first if you want a clean download, then re-run.
  pause
  exit /b 0
)

echo Downloading OpenVINO 2026.1 runtime archive (a few hundred MB)...
curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.1/windows/openvino_toolkit_windows_2026.1.0.21367.63e31528c62_x86_64.zip --output ov.zip
if errorlevel 1 (
  echo.
  echo Download failed. Check your connection and try again.
  pause
  exit /b 1
)

echo Extracting...
tar -xf ov.zip
if errorlevel 1 (
  echo.
  echo Extract failed.
  pause
  exit /b 1
)

ren openvino_toolkit_windows_2026.1.0.21367.63e31528c62_x86_64 openvino
del ov.zip

echo.
echo Done. OpenVINO runtime is in: %cd%\openvino
echo You can close this window.
pause
