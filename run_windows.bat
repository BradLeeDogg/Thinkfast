@echo off
setlocal
cd /d "%~dp0"
title agentic-ai

echo ==========================================================
echo    agentic-ai  -  setup and launch (Windows, fast/Ollama)
echo ==========================================================
echo.

REM --- 1) Make sure Python is installed ------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY goto :need_python
echo Python: %PY%

REM --- 2) Make sure Ollama (the fast engine) is installed ------------------
where ollama >nul 2>nul
if not errorlevel 1 goto :have_ollama
echo Ollama is not installed. Trying to install it automatically...
where winget >nul 2>nul
if errorlevel 1 goto :ollama_manual
winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
echo.
echo Ollama was installed. Please double-click this file again so it can be
echo found (wait a few seconds first).
goto :bye
:have_ollama
echo Ollama: found

REM --- 3) Download the AI model (quantized, fast) -------------------------
set "AGENT_MODEL=qwen2.5:1.5b"
echo Making sure the model %AGENT_MODEL% is downloaded (first time only)...
ollama pull %AGENT_MODEL%
if errorlevel 1 goto :ollama_not_running

REM --- 4) Install this app (light - no giant ML libraries) ----------------
if not exist ".venv\Scripts\python.exe" (
  echo Creating the environment ^(first run only^)...
  %PY% -m venv .venv
  if errorlevel 1 goto :error
)
echo Installing the app...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e ".[ollama,webui]"
if errorlevel 1 goto :error

REM --- 5) Launch -----------------------------------------------------------
set "AGENT_BACKEND=ollama"
echo.
echo ==========================================================
echo  Starting. Your browser opens automatically when ready.
echo  To STOP the app later, just close this black window.
echo ==========================================================
echo.
".venv\Scripts\agent-web.exe" --open
goto :end

:need_python
echo Python is not installed yet.
echo   1. I am opening the Python download page in your browser.
echo   2. Run the installer and TICK "Add python.exe to PATH".
echo   3. When it finishes, double-click this file again.
start "" "https://www.python.org/downloads/"
goto :bye

:ollama_manual
echo Could not auto-install Ollama. Opening the download page.
echo Install Ollama, then double-click this file again.
start "" "https://ollama.com/download"
goto :bye

:ollama_not_running
echo.
echo Could not download the model. Make sure the Ollama app is running
echo (look for its llama icon near the clock, bottom-right), then run this
echo file again.
goto :bye

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
exit /b 0

:bye
echo.
echo Press any key to close this window.
pause >nul
exit /b 0
