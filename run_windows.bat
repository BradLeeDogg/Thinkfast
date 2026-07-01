@echo off
setlocal
cd /d "%~dp0"
title agentic-ai

echo ==========================================================
echo    agentic-ai  -  setup and launch (Windows)
echo ==========================================================
echo.

REM --- 1) Make sure Python is installed ------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python is not installed yet.
  echo.
  echo   1. I am opening the Python download page in your browser.
  echo   2. Download it and run the installer.
  echo   3. IMPORTANT: on the FIRST installer screen, tick the box
  echo      "Add python.exe to PATH", then click "Install Now".
  echo   4. When it finishes, double-click this file again.
  echo.
  start "" "https://www.python.org/downloads/"
  echo Press any key to close this window.
  pause >nul
  exit /b 0
)
echo Using Python: %PY%
echo.

REM --- 2) Create the environment on first run ------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo Creating the environment ^(first run only^)...
  %PY% -m venv .venv
  if errorlevel 1 goto :error
)

REM --- 3) Install the app --------------------------------------------------
echo Installing the app. The first time, this can take several minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e ".[webui]"
if errorlevel 1 goto :error

REM --- 4) Pick a small model that runs on a CPU / integrated GPU -----------
REM   Too slow? Change 1.5B to 0.5B below. Real NVIDIA GPU? Try 7B.
set "AGENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct"

REM --- 5) Launch -----------------------------------------------------------
echo.
echo ==========================================================
echo  Starting up. The FIRST time it downloads the AI model
echo  ^(a couple of GB^) - you will see progress bars, be patient.
echo  Your browser opens automatically when it is ready.
echo  To STOP the app later, just close this black window.
echo ==========================================================
echo.
".venv\Scripts\agent-web.exe" --open
goto :end

:error
echo.
echo -----------------------------------------------------------
echo  Something went wrong above. Copy the error text in this
echo  window and send it to me - I will tell you what to do.
echo -----------------------------------------------------------

:end
echo.
echo (You can close this window now.)
pause >nul
