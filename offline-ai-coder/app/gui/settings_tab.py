"""Settings tab: every persisted option, validated, with live preview.

Saving writes through :func:`app.config.save_settings` (which validates
and writes atomically). Theme and font changes re-apply immediately so
the user sees what they picked; the rest take effect at once because the
main window rebuilds its Ollama client on save.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..config import describe_environment, reset_settings, save_settings
from ..errors import ConfigError
from ..logstore import log_event
from ..themes import apply_theme, get_palette


class SettingsTab:
    def __init__(self, app, frame: ttk.Frame) -> None:
        self.app = app
        self.frame = frame
        self.vars: dict[str, tk.Variable] = {}
        self._build()
        self.load_into_widgets()

    # -- widget construction ---------------------------------------------------
    def _row(self, parent, label: str, key: str, widget: str = "entry",
             width: int = 44, values: list[str] | None = None) -> tk.Variable:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, style="Panel.TLabel", width=30).pack(side="left")
        if widget == "check":
            var: tk.Variable = tk.BooleanVar()
            ttk.Checkbutton(row, variable=var).pack(side="left")
        elif widget == "combo":
            var = tk.StringVar()
            ttk.Combobox(row, textvariable=var, values=values or [],
                         state="readonly", width=width - 18).pack(side="left")
        elif widget == "spin":
            var = tk.IntVar()
            ttk.Spinbox(row, textvariable=var, from_=8, to=20, width=6).pack(side="left")
        else:
            var = tk.StringVar()
            ttk.Entry(row, textvariable=var, width=width).pack(side="left", fill="x",
                                                              expand=True)
        self.vars[key] = var
        return var

    def _build(self) -> None:
        app = self.app
        left = ttk.Frame(self.frame)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=6)

        server = ttk.Labelframe(left, text="Ollama server")
        server.pack(fill="x", pady=4)
        self._row(server, "Server address", "ollama_base_url")
        self._row(server, "Main model", "main_model")
        self._row(server, "Fast model", "fast_model")
        self._row(server, "Chat uses", "default_model_role", "combo",
                  values=["main", "fast"])
        self._row(server, "Quick-request timeout (s)", "request_timeout_seconds")
        self._row(server, "Chat timeout (s)", "chat_timeout_seconds")
        self._row(server, "Check server at startup", "check_server_on_start", "check")

        chat = ttk.Labelframe(left, text="Chat behaviour")
        chat.pack(fill="x", pady=4)
        self._row(chat, "Temperature (0 = careful, 1 = wild)", "temperature")
        self._row(chat, "Context size (tokens)", "num_ctx")

        look = ttk.Labelframe(left, text="Look and feel")
        look.pack(fill="x", pady=4)
        self._row(look, "Theme", "theme", "combo", values=["dark", "light"])
        self._row(look, "Font size", "font_size", "spin")

        runbox = ttk.Labelframe(left, text="Files and runner")
        runbox.pack(fill="x", pady=4)
        self._row(runbox, "Runner timeout (s)", "runner_timeout_seconds")
        self._row(runbox, "Max file preview (bytes)", "max_file_preview_bytes")
        self._row(runbox, "Ask before deleting a model",
                  "confirm_before_delete_model", "check")

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=8)
        ttk.Button(buttons, text="Save settings", style="Accent.TButton",
                   command=self.save).pack(side="left")
        ttk.Button(buttons, text="Reload from disk",
                   command=self.load_into_widgets).pack(side="left", padx=4)
        ttk.Button(buttons, text="Reset to defaults",
                   command=self.reset).pack(side="left", padx=4)
        self.message_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.message_var,
                  style="Dim.TLabel", wraplength=640).pack(anchor="w")

        right = ttk.Labelframe(self.frame, text="Environment")
        right.pack(side="left", fill="y", padx=(0, 8), pady=6)
        env = describe_environment()
        text = "\n".join(f"{k}: {v}" for k, v in env.items())
        label = ttk.Label(right, text=text, style="Panel.TLabel", justify="left")
        label.pack(padx=8, pady=8, anchor="nw")

    # -- load/save ---------------------------------------------------------------
    _FLOAT_KEYS = {"request_timeout_seconds", "chat_timeout_seconds", "temperature",
                   "runner_timeout_seconds"}
    _INT_KEYS = {"num_ctx", "font_size", "max_file_preview_bytes"}
    _BOOL_KEYS = {"confirm_before_delete_model", "check_server_on_start"}

    def load_into_widgets(self) -> None:
        s = self.app.settings
        values = {
            "ollama_base_url": s.ollama_base_url,
            "main_model": s.main_model,
            "fast_model": s.fast_model,
            "default_model_role": s.default_model_role,
            "request_timeout_seconds": s.request_timeout_seconds,
            "chat_timeout_seconds": s.chat_timeout_seconds,
            "temperature": s.temperature,
            "num_ctx": s.num_ctx,
            "theme": s.theme,
            "font_size": s.font_size,
            "runner_timeout_seconds": s.runner_timeout_seconds,
            "max_file_preview_bytes": s.max_file_preview_bytes,
            "confirm_before_delete_model": s.confirm_before_delete_model,
            "check_server_on_start": s.check_server_on_start,
        }
        for key, value in values.items():
            var = self.vars.get(key)
            if var is not None:
                var.set(value)
        self.message_var.set("Settings loaded from disk.")

    def _collect(self) -> dict:
        collected: dict[str, object] = {}
        for key, var in self.vars.items():
            raw = var.get()
            if key in self._BOOL_KEYS:
                collected[key] = bool(raw)
            elif key in self._INT_KEYS:
                collected[key] = int(raw)
            elif key in self._FLOAT_KEYS:
                collected[key] = float(raw)
            else:
                collected[key] = str(raw).strip()
        return collected

    def save(self) -> None:
        app = self.app
        try:
            updates = self._collect()
        except (ValueError, tk.TclError):
            self.message_var.set("Some fields are not valid numbers - check them and save again.")
            return
        for key, value in updates.items():
            setattr(app.settings, key, value)
        try:
            save_settings(app.settings)
        except ConfigError as exc:
            self.message_var.set(str(exc))
            app.show_error(exc)
            return
        app.rebuild_client()
        app.palette = get_palette(app.settings.theme)
        apply_theme(app.root, app.palette, app.settings.font_size)
        self.message_var.set("Saved. Theme/font applied; the new server settings are live.")
        app.set_status("Settings saved.")
        log_event("settings saved")

    def reset(self) -> None:
        if not self.app.ask_yes_no("Reset settings",
                                   "Put every setting back to the defaults?"):
            return
        self.app.settings = reset_settings()
        self.app.rebuild_client()
        self.load_into_widgets()
        self.message_var.set("Defaults restored and saved.")
        log_event("settings reset to defaults")


def build(notebook, app) -> ttk.Frame:
    frame = ttk.Frame(notebook)
    SettingsTab(app, frame)
    return frame
