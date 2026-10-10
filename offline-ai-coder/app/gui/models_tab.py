"""Models tab: see, download, inspect, and remove local Ollama models."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..logstore import log_event
from ..ollama_client import ModelInfo
from ..themes import text_widget_kwargs

SUGGESTED_MODELS = [
    "qwen2.5-coder:7b",
    "qwen2.5-coder:14b",
    "qwen2.5-coder:32b",
    "llama3.1:8b",
    "deepseek-coder-v2:16b",
]


class ModelsTab:
    def __init__(self, app, frame: ttk.Frame) -> None:
        self.app = app
        self.frame = frame
        self.models: list[ModelInfo] = []
        self._build()

    def _build(self) -> None:
        app = self.app
        toolbar = ttk.Frame(self.frame)
        toolbar.pack(fill="x", padx=8, pady=6)
        ttk.Button(toolbar, text="Refresh list", command=self.refresh_list).pack(side="left")
        ttk.Button(toolbar, text="Details", command=self.show_details).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Delete selected", command=self.delete_selected).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Use as Main", command=lambda: self.set_role("main")).pack(side="left", padx=(12, 4))
        ttk.Button(toolbar, text="Use as Fast", command=lambda: self.set_role("fast")).pack(side="left")

        columns = ("name", "size", "family", "params", "quant", "tag")
        self.tree = ttk.Treeview(self.frame, columns=columns, show="headings", height=12)
        for col, title, width in [
            ("name", "Model", 260), ("size", "Size", 90), ("family", "Family", 110),
            ("params", "Params", 80), ("quant", "Quant", 80), ("tag", "Tag", 90),
        ]:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=8)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        self.role_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.role_var, style="Dim.TLabel").pack(
            anchor="w", padx=8, pady=(4, 0))
        self._update_role_label()

        pull_card = ttk.Labelframe(self.frame, text="Download a model (one-time, needs internet)")
        pull_card.pack(fill="x", padx=8, pady=8)
        row = ttk.Frame(pull_card)
        row.pack(fill="x", padx=8, pady=6)
        ttk.Label(row, text="Model name:", style="Panel.TLabel").pack(side="left")
        self.pull_var = tk.StringVar(value=SUGGESTED_MODELS[0])
        combo = ttk.Combobox(row, textvariable=self.pull_var, values=SUGGESTED_MODELS,
                             width=30)
        combo.pack(side="left", padx=6)
        ttk.Button(row, text="Download", style="Accent.TButton",
                   command=self.pull_model).pack(side="left", padx=6)
        self.pull_status = tk.StringVar(value="Downloads are big - the 14B is about 9 GB.")
        ttk.Label(row, textvariable=self.pull_status, style="Panel.TLabel").pack(side="left", padx=6)
        self.progress = ttk.Progressbar(pull_card, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", padx=8, pady=(0, 8))

        self.details = tk.Text(self.frame, height=7, state="disabled",
                               **text_widget_kwargs(app.palette, code=True,
                                                    font_size=app.settings.font_size))
        self.details.pack(fill="x", padx=8, pady=(0, 8))

    # -- helpers ---------------------------------------------------------------
    def _update_role_label(self) -> None:
        s = self.app.settings
        self.role_var.set(f"Main model: {s.main_model}    Fast model: {s.fast_model}    "
                          f"(chat uses: {s.active_model})")

    def _selected(self) -> ModelInfo | None:
        selection = self.tree.selection()
        if not selection:
            return None
        index = int(selection[0])
        return self.models[index] if 0 <= index < len(self.models) else None

    def _on_select(self, _event=None) -> None:
        model = self._selected()
        if model:
            self.pull_var.set(model.name)

    def _set_details(self, text: str) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    # -- actions -----------------------------------------------------------------
    def refresh_list(self) -> None:
        app = self.app
        app.set_status("Asking Ollama for the model list...")

        def work():
            return app.client.list_models()

        def done(models: list[ModelInfo]) -> None:
            self.models = models
            for item in self.tree.get_children():
                self.tree.delete(item)
            for index, model in enumerate(models):
                self.tree.insert("", "end", iid=str(index), values=(
                    model.name, model.size_human, model.family,
                    model.parameter_size, model.quantization, model.tag,
                ))
            app.set_status(f"{len(models)} model(s) installed.")
            app.set_server_status(f"Ollama: on ({len(models)} models)")
            log_event(f"model list refreshed: {len(models)}")

        app.run_async(work, done, app.show_error)

    def show_details(self) -> None:
        model = self._selected()
        if model is None:
            self.app.show_info("Models", "Pick a model in the list first.")
            return
        app = self.app

        def work():
            return app.client.show_model(model.name)

        def done(info: dict) -> None:
            lines = [f"{model.name} - {model.size_human}"]
            for key in ("modelfile", "parameters", "template"):
                if info.get(key):
                    lines.append("")
                    lines.append(f"--- {key} ---")
                    lines.append(str(info[key])[:4000])
            details = info.get("details")
            if isinstance(details, dict):
                lines.append("")
                lines.append("Details: " + ", ".join(f"{k}={v}" for k, v in details.items()))
            self._set_details("\n".join(lines))

        app.run_async(work, done, app.show_error)

    def delete_selected(self) -> None:
        model = self._selected()
        if model is None:
            self.app.show_info("Models", "Pick a model in the list first.")
            return
        app = self.app
        if app.settings.confirm_before_delete_model:
            if not app.ask_yes_no(
                "Delete model",
                f"Delete '{model.name}' ({model.size_human}) from this PC?\n\n"
                "You can download it again later, but it needs internet and time.",
            ):
                return

        def work():
            app.client.delete_model(model.name)
            return model.name

        def done(name: str) -> None:
            log_event(f"model deleted: {name}")
            app.set_status(f"Deleted {name}.")
            self.refresh_list()

        app.run_async(work, done, app.show_error)

    def set_role(self, role: str) -> None:
        model = self._selected()
        if model is None:
            self.app.show_info("Models", "Pick a model in the list first.")
            return
        settings = self.app.settings
        if role == "main":
            settings.main_model = model.name
        else:
            settings.fast_model = model.name
        self.app.save_settings_safely()
        self._update_role_label()
        self.app.set_status(f"'{model.name}' is now the {role} model.")
        log_event(f"model role set: {role}={model.name}")

    def pull_model(self) -> None:
        name = self.pull_var.get().strip()
        if not name:
            self.app.show_info("Models", "Type or pick a model name first.")
            return
        app = self.app
        app.set_status(f"Downloading {name}...")
        self.pull_status.set("Starting download...")
        self.progress["value"] = 0

        def work():
            last_message = [""]

            def progress(message: str, fraction) -> None:
                last_message[0] = message
                # Progress UI is updated by polling last_message via after()
                app.root.after(0, lambda: self._pull_progress(message, fraction))

            return app.client.pull_model(name, progress=progress)

        def done(last: str) -> None:
            self.pull_status.set(f"Done: {last}")
            self.progress["value"] = 1000
            app.set_status(f"Model {name} is ready.")
            log_event(f"model pulled: {name}")
            self.refresh_list()

        def failed(exc) -> None:
            self.pull_status.set("Download stopped - see the message.")
            app.show_error(exc)

        app.run_async(work, done, failed)

    def _pull_progress(self, message: str, fraction) -> None:
        self.pull_status.set(message)
        if fraction is not None:
            self.progress["value"] = int(fraction * 1000)


_tab: ModelsTab | None = None


def build(notebook, app) -> ttk.Frame:
    global _tab
    frame = ttk.Frame(notebook)
    _tab = ModelsTab(app, frame)
    return frame


def refresh(app) -> None:
    if _tab is not None and not _tab.models:
        _tab.refresh_list()
