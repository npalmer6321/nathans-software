# Offline AI Coder

Set up a free, fully offline AI coding assistant on a Windows PC — then run
it day-to-day from a friendly desktop app. No accounts, no cloud, no
subscriptions, and no cost after the one-time model download.

## Two halves

1. **The installer** (PowerShell + NSIS) puts everything on the machine
   once: Ollama, the coder models, VS Code + Cline, Python + Aider, and a
   practice project.
2. **The desktop app** (this folder's `main.py`) is your control center
   afterwards: check the rig, manage models, chat with the local model,
   browse your project, and run your tests — all without a terminal.

## The desktop app

Pure Python standard library (Tkinter GUI) — nothing to `pip install`.

**Run it**

```
python main.py
```

(Windows, macOS, and Linux all work. Needs Python 3.9+ and, for the AI
features, a local Ollama server with a model pulled.)

**Tabs**

- **Dashboard** — is Ollama answering, what shape is this PC in, one-click
  full readiness check with the most useful next steps
- **Models** — list / download / inspect / delete local models, pick the
  Main and Fast models
- **Chat** — streaming chat with the local model; optional "include my
  project files" context; sessions you can save, reopen, and copy
- **Prompts** — a built-in library of coding prompt templates
  (explain, debug, tests, review...) with fill-in blanks, plus your own
- **Files** — browse a project folder, preview files, send code to Chat
- **Run & Test** — run unit tests or a Python file by button click (no
  shell, no free-typed commands — the exact command is shown and approved
  first), then ask the AI about the output
- **Setup** — the full setup checklist plus the installer's own setup log
- **Logs** — view/search the app's log, errors only if you want
- **Settings** — server address, models, chat behaviour, theme, timeouts;
  validated and saved atomically

App data (settings, chats, prompts, log) lives in the per-user folder —
on Windows: `%LOCALAPPDATA%\OfflineCodingAgent\`.

**Tests** (non-GUI logic; run from this folder):

```
python -m unittest discover -s tests -v
```

## The installer

- Installs Ollama and pulls Qwen2.5-Coder (14B main + 7B fast), creates a 16K-context model
- Installs VS Code + the Cline extension and Python + Aider, pre-pointed at the local model
- Copies a practice project to `%USERPROFILE%\OfflineAgentTest`
- Adds desktop shortcuts: "Aider (Offline Coder)" and "Run Offline Test" (the airplane-mode proof)

**Needs**: Windows 10/11, ~15 GB free space, one internet connection for the first setup run. After that it works 100% offline. 16 GB RAM recommended (CPU-only is fine).

**Installer files**
- `Setup-OfflineAgent.ps1` — the silent setup script
- `Run-OfflineTest.ps1` — proves it codes with the internet off
- `Aider-Offline.bat` — launches the terminal coder
- `Start-Here.txt` — the 2-minute Cline finishing steps (read this)
- `OfflineCodingAgent.nsi` — NSIS source for the one-click installer .exe

Setup log lands at `%LOCALAPPDATA%\OfflineCodingAgent\setup.log`.

**Layout**: `app/` is the application package (`app/gui/` the Tkinter
tabs, everything else is UI-free logic under unit test), `tests/` the
test suite, `sample/` the practice project the installer copies.
