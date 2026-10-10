"""Logs tab: view, search, and copy the app log; jump to recent errors."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .. import logstore
from ..themes import text_widget_kwargs


class LogsTab:
    def __init__(self, app, frame: ttk.Frame) -> None:
        self.app = app
        self.frame = frame
        self.lines: list[logstore.LogLine] = []
        self._build()

    def _build(self) -> None:
        app = self.app
        top = ttk.Frame(self.frame)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="Reload", style="Accent.TButton",
                   command=self.reload).pack(side="left")
        ttk.Label(top, text="Search:", style="Dim.TLabel").pack(side="left", padx=(10, 0))
        self.search_var = tk.StringVar()
        entry = ttk.Entry(top, textvariable=self.search_var, width=30)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda _e: self.apply_filter())
        ttk.Button(top, text="Find", command=self.apply_filter).pack(side="left")
        self.errors_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Errors only", variable=self.errors_only,
                        command=self.apply_filter).pack(side="left", padx=8)
        ttk.Button(top, text="Copy visible", command=self.copy_visible).pack(side="right")
        ttk.Button(top, text="Clear view", command=self.clear).pack(side="right", padx=4)

        self.count_var = tk.StringVar(value="")
        ttk.Label(self.frame, textvariable=self.count_var,
                  style="Dim.TLabel").pack(anchor="w", padx=8)
        self.text = tk.Text(self.frame, state="disabled",
                            **text_widget_kwargs(app.palette, code=True,
                                                 font_size=app.settings.font_size))
        self.text.pack(fill="both", expand=True, padx=8, pady=6)
        self.text.tag_configure("err", foreground=app.palette.bad)

    def reload(self) -> None:
        app = self.app

        def work():
            return logstore.read_log()

        def done(lines) -> None:
            self.lines = lines
            self.apply_filter()
            app.set_status(f"Log reloaded ({len(lines)} lines).")

        app.run_async(work, done, app.show_error)

    def apply_filter(self) -> None:
        shown = logstore.search_log(self.lines, self.search_var.get(),
                                    errors_only=self.errors_only.get())
        self.count_var.set(f"Showing {len(shown)} of {len(self.lines)} log lines "
                           f"from {logstore.log_path()}")
        widget = self.text
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        for line in shown:
            start = widget.index("end")
            widget.insert("end", line.raw + "\n")
            if line.is_error:
                widget.tag_add("err", start, widget.index("end"))
        widget.configure(state="disabled")
        widget.see("end")

    def copy_visible(self) -> None:
        self.app.root.clipboard_clear()
        self.app.root.clipboard_append(self.text.get("1.0", "end"))
        self.app.set_status("Visible log copied.")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


def build(notebook, app) -> ttk.Frame:
    frame = ttk.Frame(notebook)
    LogsTab(app, frame)
    return frame


def refresh(app) -> None:
    pass
