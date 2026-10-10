"""Setup tab: full readiness checklist plus the installer's setup log.

The checklist comes from :mod:`app.checklist`; this tab also surfaces the
log the PowerShell installer wrote (``setup.log`` next to the app data)
and plain next-step guidance so nobody has to read raw checker output.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from .. import checklist as checklist_mod
from ..config import user_data_dir
from ..logstore import log_event
from ..themes import status_color, text_widget_kwargs

SETUP_LOG_NAME = "setup.log"


class SetupTab:
    def __init__(self, app, frame: ttk.Frame) -> None:
        self.app = app
        self.frame = frame
        self._build()

    def _build(self) -> None:
        app = self.app
        top = ttk.Frame(self.frame)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Is my offline rig ready?", style="H2.TLabel").pack(side="left")
        ttk.Button(top, text="Run the check", style="Accent.TButton",
                   command=self.run_check).pack(side="right")
        ttk.Button(top, text="Open data folder",
                   command=self.open_data_folder).pack(side="right", padx=4)

        self.progress = ttk.Progressbar(self.frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=8, pady=(0, 4))
        header = ("Status", "Check", "Detail")
        self.tree = ttk.Treeview(self.frame, columns=header, show="headings", height=9)
        for col, width in zip(header, (90, 240, 560)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="x", padx=8)
        self.tree.tag_configure("ok", foreground=app.palette.good)
        self.tree.tag_configure("warn", foreground=app.palette.warn)
        self.tree.tag_configure("fail", foreground=app.palette.bad)
        self.tree.tag_configure("unknown", foreground=app.palette.fg_dim)

        self.advice = tk.Text(self.frame, height=5, state="disabled",
                              **text_widget_kwargs(app.palette,
                                                   font_size=app.settings.font_size))
        self.advice.pack(fill="x", padx=8, pady=6)

        log_card = ttk.Labelframe(self.frame, text="Installer setup log (from Setup-OfflineAgent.ps1)")
        log_card.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_text = tk.Text(log_card, state="disabled",
                                **text_widget_kwargs(app.palette, code=True,
                                                     font_size=app.settings.font_size))
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Button(log_card, text="Reload setup log",
                   command=self.load_setup_log).pack(anchor="e", padx=6, pady=(0, 6))
        self.load_setup_log()

    # -- checklist ---------------------------------------------------------------
    def run_check(self) -> None:
        app = self.app
        app.set_status("Checking your setup...")

        def work():
            return checklist_mod.run_checklist(
                app.client, app.settings.main_model, app.settings.fast_model,
                app.current_project_path or None,
            )

        def done(report) -> None:
            self.progress["value"] = int(round(report.score * 100))
            for item in self.tree.get_children():
                self.tree.delete(item)
            for check in report.items:
                self.tree.insert("", "end",
                                 values=(check.status.symbol, check.title, check.detail),
                                 tags=(check.status.value,))
            steps = checklist_mod.next_steps(report, limit=5)
            advice = ("Next steps:\n" + "\n".join(f"- {s}" for s in steps)) if steps \
                else "Everything looks good - you are ready to code offline."
            self._set(self.advice, advice)
            app.set_status("Setup check finished.")
            log_event(f"setup tab check: score={report.score:.2f}")

        app.run_async(work, done, app.show_error)

    # -- setup log ------------------------------------------------------------------
    def setup_log_path(self) -> Path:
        return user_data_dir() / SETUP_LOG_NAME

    def load_setup_log(self) -> None:
        path = self.setup_log_path()
        if not path.exists():
            self._set(self.log_text,
                      f"No installer log found at {path}.\n"
                      "It appears after you run Setup-OfflineAgent.ps1 (or the .exe installer).")
            return

        def work():
            text = path.read_text(encoding="utf-8", errors="replace")
            return "\n".join(text.splitlines()[-400:])

        self.app.run_async(work, lambda text: self._set(self.log_text, text),
                           self.app.show_error)

    def _set(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")
        widget.see("end")

    # -- misc ----------------------------------------------------------------------
    def open_data_folder(self) -> None:
        folder = str(user_data_dir())
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", folder], check=False)
            else:
                subprocess.run(["xdg-open", folder], check=False)
            self.app.set_status(f"Opened {folder}")
        except Exception as exc:
            self.app.show_error(exc)


def build(notebook, app) -> ttk.Frame:
    frame = ttk.Frame(notebook)
    SetupTab(app, frame)
    return frame


def status_tag_for(value: str, palette) -> str:
    return status_color(value, palette)
