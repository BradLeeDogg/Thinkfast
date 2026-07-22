@echo off
setlocal
cd /d "%~dp0"
title agentic-ai - index your library

echo ==========================================================
echo   agentic-ai  -  make your AI an expert on your PDFs
echo ==========================================================
echo.
echo This reads your PDF books once and builds a search index, so the
echo AI can look things up and quote them. Plug in the drive with your
echo PDFs before you start.
echo.

set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY goto :need_base

where ollama >nul 2>nul
if errorlevel 1 goto :need_base

echo Downloading the search model (first time only)...
ollama pull nomic-embed-text
if errorlevel 1 goto :ollama_off

if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv
)
echo Installing the library tools (first time only)...
".venv\Scripts\python.exe" -m pip install -q --upgrade pip
".venv\Scripts\python.exe" -m pip install -q -e ".[ollama,webui,library]"
if errorlevel 1 goto :error

echo.
echo Paste the folder that contains your PDFs, then press Enter.
echo   Tip: open the folder in File Explorer, click the address bar at
echo   the top, copy it, then right-click in this window to paste.
echo   Example:  E:\Novels
echo.
set /p FOLDER=Folder:
if "%FOLDER%"=="" goto :no_folder
if not exist "%FOLDER%" goto :no_folder

echo.
echo Indexing "%FOLDER%" ... a big library can take a few hours.
echo You can leave this running; it saves as it goes and can resume.
echo.
".venv\Scripts\agent-index.exe" "%FOLDER%"
if errorlevel 1 goto :error

echo.
echo ==========================================================
echo  Done! Now double-click run_windows.bat and ask about your
echo  books, for example:
echo    "Search my library for passages about the sea"
echo    "What happens at the end of Moby Dick?"
echo ==========================================================
goto :end

:need_base
echo Please run run_windows.bat first (it sets up Python and Ollama),
echo then come back and run this file.
goto :end

:ollama_off
echo Could not download the search model. Make sure the Ollama app is
echo running (llama icon near the clock, bottom-right), then try again.
goto :end

:no_folder
echo That folder was not found. Please run this file again and paste the
echo correct folder path.
goto :end

:error
echo.
echo Something went wrong above. Copy the text in this window and send
echo it to me - I will tell you what to do.

:end
echo.
echo (You can close this window now.)
pause >nul
