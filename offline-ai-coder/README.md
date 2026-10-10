# Offline AI Coder

Set up a free, fully offline AI coding assistant on a Windows PC.

**What it does**
- Installs Ollama and pulls Qwen2.5-Coder (14B main + 7B fast), creates a 16K-context model
- Installs VS Code + the Cline extension and Python + Aider, pre-pointed at the local model
- Copies a practice project to `%USERPROFILE%\OfflineAgentTest`
- Adds desktop shortcuts: "Aider (Offline Coder)" and "Run Offline Test" (the airplane-mode proof)

**Needs**: Windows 10/11, ~15 GB free space, one internet connection for the first setup run. After that it works 100% offline. 16 GB RAM recommended (CPU-only is fine).

**Files**
- `Setup-OfflineAgent.ps1` — the silent setup script
- `Run-OfflineTest.ps1` — proves it codes with the internet off
- `Aider-Offline.bat` — launches the terminal coder
- `Start-Here.txt` — the 2-minute Cline finishing steps (read this)
- `OfflineCodingAgent.nsi` — NSIS source for the one-click installer .exe

Setup log lands at `%LOCALAPPDATA%\OfflineCodingAgent\setup.log`.
