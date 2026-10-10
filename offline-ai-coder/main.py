#!/usr/bin/env python3
"""Offline AI Coder - desktop control center for your local AI coding rig.

Run:
    python main.py

Pure Python standard library GUI (Tkinter). Talks to a local Ollama
server; everything works offline once your models are downloaded.
"""

from app.gui import run

if __name__ == "__main__":
    run()
