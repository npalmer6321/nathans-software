# Nathans Software

Free tools built in Texas by a construction PM who codes at night. Everything here is original, works offline where it says it does, and costs nothing.

Follow the builds on Facebook: **Nathans Software** — https://www.facebook.com/profile.php?id=61590507882825

## The programs

### 1. Offline AI Coder (`offline-ai-coder/`)
Your PC can run its own AI — no internet, no subscription, nothing leaving your house.
One installer sets up Ollama + Qwen2.5-Coder on Windows, wires it into VS Code (Cline) and Aider, and drops a test project on your desktop so you can flip on airplane mode and watch it keep coding.
Start with `Start-Here.txt`. Installer source: `OfflineCodingAgent.nsi`, setup logic: `Setup-OfflineAgent.ps1`.

### 2. SelfForge (`selfforge/`)
A tiny neural network, written from scratch in Kotlin (no frameworks), that trains itself.
It starts from a handful of hand-typed examples, practices on its own output, keeps what works, throws out what doesn't, and restores its best snapshot if it ever gets worse. From 8 examples it reached 91% on patterns it had never seen.
Includes the core engine, an Android app, and a PC version.

### 3. ForgeGPT (`forgegpt/`)
A transformer language model — the same architecture family as ChatGPT, in miniature — written from scratch in pure Kotlin. Every matrix multiply, attention head, and backprop step is hand-coded and gradient-checked.
Trained on public-domain books (Alice in Wonderland, Pride and Prejudice, Sherlock Holmes, Frankenstein, Dickens, Moby Dick), it writes recognizably English sentences in those voices. It rambles — that's the honest ceiling at this size — but it's real.
Run it: `java -jar forgegpt.jar` (needs Java 17+). Retrain it with `build.sh` and the included corpus.

## License
Built by Nathan Lee Palmer. Free to use and learn from — please credit Nathans Software if you share it.
