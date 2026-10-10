"""Main window: owns shared state and services, hosts the tab panels.

Tabs are plain modules with ``build(notebook, app) -> frame`` and an
optional ``refresh(app)`` / ``on_close(app)``. Everything slow (network,
subprocess) runs on worker threads; results come back through a queue
polled with ``after`` so the UI never freezes.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .. import __version__
from ..config import LoadResult, Settings, load_settings, save_settings
from ..errors import AppError
from ..logstore import log_error, log_event
from ..ollama_client import OllamaClient
from ..themes import apply_theme, get_palette
from . import (chat_tab, dashboard_tab, files_tab, logs_tab, models_tab,
               prompts_tab, runner_tab, settings_tab, setup_tab)

TAB_BUILDERS = [
    ("Dashboard", dashboard_tab),
    ("Models", models_tab),
    ("Chat", chat_tab),
    ("Prompts", prompts_tab),
    ("Files", files_tab),
    ("Run & Test", runner_tab),
    ("Setup", setup_tab),
    ("Logs", logs_tab),
    ("Settings", settings_tab),
]


class MainWindow:
    """Application object passed to every tab as ``app``."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.load_result: LoadResult = load_settings()
        self.settings: Settings = self.load_result.settings
        self.palette = get_palette(self.settings.theme)
        self.client = OllamaClient(
            self.settings.ollama_base_url,
            timeout=self.settings.request_timeout_seconds,
        )
        self.current_project_path: str = self.settings.last_project_dir
        self._jobs: "queue.Queue[tuple]" = queue.Queue()
        self._tabs: dict[str, object] = {}
        self._closing = False

        root.title(f"Offline AI Coder {__version__}")
        root.geometry("1180x760")
        root.minsize(940, 600)
        apply_theme(root, self.palette, self.settings.font_size)
        self._build_menu()
        self._build_layout()
        self._poll_jobs()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        if self.load_result.warning:
            self.set_status(self.load_result.warning)
            log_event("settings warning: " + self.load_result.warning)
        elif self.load_result.created_new:
            self.set_status("Welcome! First run - check the Setup tab to verify your rig.")
        else:
            self.set_status("Ready.")
        log_event(f"app started (theme={self.settings.theme})")
        if self.settings.check_server_on_start:
            dashboard_tab.kick_initial_check(self)

    # -- layout ------------------------------------------------------------
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open project folder...",
                              command=lambda: self.select_tab("Files"))
        file_menu.add_command(label="Open Settings",
                              command=lambda: self.select_tab("Settings"))
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        go_menu = tk.Menu(menubar, tearoff=0)
        for title, _ in TAB_BUILDERS:
            go_menu.add_command(label=title,
                                command=lambda t=title: self.select_tab(t))
        menubar.add_cascade(label="Go", menu=go_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About Offline AI Coder", command=self._about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    def _build_layout(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        for title, module in TAB_BUILDERS:
            frame = module.build(self.notebook, self)
            self.notebook.add(frame, text=title)
            self._tabs[title] = module
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=6, pady=4)
        self.status_var = tk.StringVar(value="Starting...")
        self.status_label = ttk.Label(bar, textvariable=self.status_var,
                                      style="Dim.TLabel")
        self.status_label.pack(side="left", fill="x", expand=True)
        self.server_var = tk.StringVar(value="Ollama: not checked yet")
        ttk.Label(bar, textvariable=self.server_var, style="Dim.TLabel").pack(side="right")

    # -- shared services -----------------------------------------------------
    def rebuild_client(self) -> None:
        """Recreate the Ollama client after settings change."""
        self.client = OllamaClient(
            self.settings.ollama_base_url,
            timeout=self.settings.request_timeout_seconds,
        )

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def set_server_status(self, text: str) -> None:
        self.server_var.set(text)

    def select_tab(self, title: str) -> None:
        for index in range(self.notebook.index("end")):
            if self.notebook.tab(index, "text") == title:
                self.notebook.select(index)
                return

    def _on_tab_changed(self, _event=None) -> None:
        current = self.notebook.tab(self.notebook.select(), "text")
        module = self._tabs.get(current)
        refresh = getattr(module, "refresh", None)
        if callable(refresh):
            try:
                refresh(self)
            except Exception as exc:  # a tab bug must not kill the app
                log_error(f"refresh failed for tab {current}", exc)

    def run_async(self, work, on_done=None, on_error=None) -> None:
        """Run ``work()`` on a thread; callbacks fire back on the UI thread."""
        def target() -> None:
            try:
                result = work()
            except Exception as exc:  # delivered to on_error on UI thread
                self._jobs.put(("error", exc, on_error))
            else:
                self._jobs.put(("done", result, on_done))

        threading.Thread(target=target, daemon=True).start()

    def _poll_jobs(self) -> None:
        try:
            while True:
                kind, payload, callback = self._jobs.get_nowait()
                if callback is None:
                    if kind == "error":
                        self.show_error(payload)
                    continue
                try:
                    callback(payload)
                except Exception as exc:
                    log_error("async callback failed", exc)
        except queue.Empty:
            pass
        if not self._closing:
            self.root.after(80, self._poll_jobs)

    # -- dialogs / lifecycle -------------------------------------------------
    def show_error(self, exc: BaseException) -> None:
        if isinstance(exc, AppError):
            message = str(exc)
        else:
            message = f"Something unexpected happened:\n\n{exc}"
        log_error("shown to user", exc if isinstance(exc, BaseException) else None)
        messagebox.showerror("Offline AI Coder", message, parent=self.root)
        self.set_status("Something went wrong - see the message or the Logs tab.")

    def show_info(self, title: str, message: str) -> None:
        messagebox.showinfo(title, message, parent=self.root)

    def ask_yes_no(self, title: str, message: str) -> bool:
        return bool(messagebox.askyesno(title, message, parent=self.root))

    def _about(self) -> None:
        self.show_info(
            "About Offline AI Coder",
            f"Offline AI Coder {__version__}\n\n"
            "A control center for running AI coding help entirely on your "
            "own PC - no accounts, no cloud, no subscriptions.\n\n"
            "Built with the Python standard library only.",
        )

    def save_settings_safely(self) -> None:
        try:
            save_settings(self.settings)
        except AppError as exc:
            self.show_error(exc)

    def on_close(self) -> None:
        self._closing = True
        for module in self._tabs.values():
            hook = getattr(module, "on_close", None)
            if callable(hook):
                try:
                    hook(self)
                except Exception as exc:
                    log_error("tab on_close failed", exc)
        self.save_settings_safely()
        log_event("app closed")
        self.root.destroy()


def run() -> None:
    """Entry point used by ``main.py`` and ``python -m app.gui``."""
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()
