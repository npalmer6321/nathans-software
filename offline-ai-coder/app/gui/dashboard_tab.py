"""Dashboard tab: is the rig on, is it healthy, what should I do next."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .. import checklist as checklist_mod
from .. import system_info
from ..logstore import log_event
from ..ollama_client import ServerStatus
from ..themes import status_color, text_widget_kwargs


class DashboardTab:
    def __init__(self, app, frame: ttk.Frame) -> None:
        self.app = app
        self.frame = frame
        self._build()

    def _build(self) -> None:
        app = self.app
        pad = {"padx": 8, "pady": 4}

        title = ttk.Label(self.frame, text="Your offline coding rig",
                          style="Title.TLabel")
        title.pack(anchor="w", **pad)

        # Server status card
        server_card = ttk.Labelframe(self.frame, text="Ollama server")
        server_card.pack(fill="x", **pad)
        self.server_text = tk.StringVar(value="Not checked yet.")
        ttk.Label(server_card, textvariable=self.server_text,
                  style="Panel.TLabel").pack(side="left", padx=8, pady=6)
        ttk.Button(server_card, text="Check now",
                   command=self.check_server).pack(side="right", padx=8, pady=6)
        ttk.Button(server_card, text="Open Models tab",
                   command=lambda: app.select_tab("Models")).pack(side="right", pady=6)

        # Machine card
        machine_card = ttk.Labelframe(self.frame, text="This PC")
        machine_card.pack(fill="x", **pad)
        self.machine_text = tk.Text(machine_card, height=6, state="disabled",
                                    **text_widget_kwargs(app.palette,
                                                         font_size=app.settings.font_size))
        self.machine_text.pack(fill="x", padx=8, pady=6)
        self._fill_machine()

        # Readiness card
        ready_card = ttk.Labelframe(self.frame, text="Readiness")
        ready_card.pack(fill="both", expand=True, **pad)
        self.progress = ttk.Progressbar(ready_card, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=8, pady=(8, 2))
        self.ready_text = tk.Text(ready_card, height=10, state="disabled",
                                  **text_widget_kwargs(app.palette,
                                                       font_size=app.settings.font_size))
        self.ready_text.pack(fill="both", expand=True, padx=8, pady=6)
        ttk.Button(ready_card, text="Run full setup check",
                   style="Accent.TButton",
                   command=self.run_checklist).pack(anchor="e", padx=8, pady=(0, 8))

        # Next steps
        steps_card = ttk.Labelframe(self.frame, text="Suggested next steps")
        steps_card.pack(fill="x", **pad)
        self.steps_var = tk.StringVar(
            value="Run a setup check and I will point at the most useful next step.")
        ttk.Label(steps_card, textvariable=self.steps_var,
                  style="Panel.TLabel", wraplength=900).pack(anchor="w", padx=8, pady=6)

    # -- data fills ----------------------------------------------------------
    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _fill_machine(self) -> None:
        snap = system_info.snapshot(self.app.current_project_path or None)
        self._set_text(self.machine_text, system_info.format_snapshot(snap))

    # -- actions ---------------------------------------------------------------
    def check_server(self) -> None:
        app = self.app
        app.set_status("Checking the Ollama server...")
        self.server_text.set("Checking...")

        def work():
            return app.client.status()

        def done(status: ServerStatus) -> None:
            if status.reachable:
                text = (f"Running - Ollama {status.version or '?'} with "
                        f"{status.model_count} model(s) installed.")
                app.set_server_status(f"Ollama: on ({status.model_count} models)")
            else:
                text = ("Not answering. Open the Ollama app (or run "
                        "'ollama serve' in a terminal), then check again.\n"
                        f"Detail: {status.detail}")
                app.set_server_status("Ollama: off")
            self.server_text.set(text)
            app.set_status("Server check finished.")
            log_event(f"server check: reachable={status.reachable}")

        app.run_async(work, done, app.show_error)

    def run_checklist(self) -> None:
        app = self.app
        app.set_status("Running the full setup check...")

        def work():
            return checklist_mod.run_checklist(
                app.client, app.settings.main_model, app.settings.fast_model,
                app.current_project_path or None,
            )

        def done(report) -> None:
            self.progress["value"] = int(round(report.score * 100))
            rendered = report.render()
            self._set_text(self.ready_text, rendered)
            steps = checklist_mod.next_steps(report)
            self.steps_var.set(
                " ".join(f"{i + 1}. {step}" for i, step in enumerate(steps))
                if steps else "Everything checks out - open the Chat tab and code."
            )
            app.set_status("Setup check finished.")
            log_event(f"checklist run: score={report.score:.2f} ready={report.ready}")

        app.run_async(work, done, app.show_error)


_tab: DashboardTab | None = None


def build(notebook, app) -> ttk.Frame:
    global _tab
    frame = ttk.Frame(notebook)
    _tab = DashboardTab(app, frame)
    return frame


def kick_initial_check(app) -> None:
    """Triggered by the main window after startup when enabled."""
    if _tab is not None:
        _tab.check_server()


def refresh(app) -> None:
    if _tab is not None:
        _tab._fill_machine()
