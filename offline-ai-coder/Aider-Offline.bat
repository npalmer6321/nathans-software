@echo off
REM Aider (offline) - terminal coding partner, runs on your local model only.
set OLLAMA_API_BASE=http://127.0.0.1:11434
cd /d %USERPROFILE%
set AIDER=%LOCALAPPDATA%\Programs\Python\Python312\Scripts\aider.exe
if exist "%AIDER%" (
    "%AIDER%" %*
) else (
    aider %*
)
pause
