"""Prompts tab: browse, fill in, edit, and reuse coding prompt templates."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..logstore import log_event
from ..prompts import PromptLibrary, variable_help
from ..themes import text_widget_kwargs


class PromptsTab:
    def __init__(self, app, frame: ttk.Frame) -> None:
        self.app = app
        self.frame = frame
        self.library = PromptLibrary.load()
        self.current_id: str | None = None
        self._build()
        self._reload_list()

    def _build(self) -> None:
        app = self.app
        left = ttk.Frame(self.frame)
        left.pack(side="left", fill="y", padx=8, pady=6)
        search_row = ttk.Frame(left)
        search_row.pack(fill="x")
        ttk.Label(search_row, text="Search:", style="Dim.TLabel").pack(side="left")
        self.search_var = tk.StringVar()
        entry = ttk.Entry(search_row, textvariable=self.search_var, width=22)
        entry.pack(side="left", padx=4)
        self.search_var.trace_add("write", lambda *_: self._reload_list())
        ttk.Label(left, text="Category:", style="Dim.TLabel").pack(anchor="w", pady=(6, 0))
        self.category_var = tk.StringVar(value="All")
        self.category_box = ttk.Combobox(left, textvariable=self.category_var,
                                         state="readonly", width=22)
        self.category_box.pack(fill="x")
        self.category_box.bind("<<ComboboxSelected>>", lambda _e: self._reload_list())
        self.listbox = tk.Listbox(left, width=34, height=24,
                                  exportselection=False)
        from ..themes import listbox_kwargs
        self.listbox.configure(**listbox_kwargs(app.palette, app.settings.font_size))
        self.listbox.pack(fill="y", expand=True, pady=6)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self._show_selected())
        self.category_box["values"] = ["All"] + self.library.categories()

        right = ttk.Frame(self.frame)
        right.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=6)
        ttk.Label(right, text="Title:", style="Dim.TLabel").pack(anchor="w")
        self.title_var = tk.StringVar()
        ttk.Entry(right, textvariable=self.title_var).pack(fill="x")
        ttk.Label(right, text="Category:", style="Dim.TLabel").pack(anchor="w", pady=(4, 0))
        self.edit_category_var = tk.StringVar()
        ttk.Entry(right, textvariable=self.edit_category_var).pack(fill="x")
        ttk.Label(right, text="Prompt text  (use {blanks} for fill-ins):",
                  style="Dim.TLabel").pack(anchor="w", pady=(4, 0))
        self.body = tk.Text(right, height=13,
                            **text_widget_kwargs(app.palette,
                                                 font_size=app.settings.font_size))
        self.body.pack(fill="both", expand=True)
        self.help_var = tk.StringVar()
        ttk.Label(right, textvariable=self.help_var, style="Dim.TLabel").pack(anchor="w", pady=2)

        ttk.Label(right, text="Fill-ins (one per line, like  language=Python):",
                  style="Dim.TLabel").pack(anchor="w", pady=(4, 0))
        self.fills = tk.Text(right, height=4,
                             **text_widget_kwargs(app.palette, code=True,
                                                  font_size=app.settings.font_size))
        self.fills.pack(fill="x")

        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=8)
        ttk.Button(buttons, text="Send to Chat", style="Accent.TButton",
                   command=self.send_to_chat).pack(side="left")
        ttk.Button(buttons, text="Copy prompt", command=self.copy_prompt).pack(side="left", padx=4)
        ttk.Button(buttons, text="Save as mine", command=self.save_as_mine).pack(side="left", padx=4)
        ttk.Button(buttons, text="Save changes", command=self.save_changes).pack(side="left", padx=4)
        ttk.Button(buttons, text="Delete", command=self.delete_current).pack(side="left", padx=4)
        ttk.Button(buttons, text="Restore built-ins", command=self.restore).pack(side="right")

    # -- list handling -----------------------------------------------------------
    def _visible(self):
        category = self.category_var.get() or "All"
        items = self.library.search(self.search_var.get())
        if category != "All":
            items = [p for p in items if p.category == category]
        return items

    def _reload_list(self) -> None:
        items = self._visible()
        self._items = items
        self.listbox.delete(0, "end")
        for prompt in items:
            marker = " *" if prompt.builtin else ""
            self.listbox.insert("end", f"{prompt.title}{marker}")
        self.category_box["values"] = ["All"] + self.library.categories()

    def _show_selected(self) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        prompt = self._items[selection[0]]
        self.current_id = prompt.id
        self.title_var.set(prompt.title)
        self.edit_category_var.set(prompt.category)
        self.body.delete("1.0", "end")
        self.body.insert("1.0", prompt.body)
        self.help_var.set(variable_help(prompt))
        prefill = "\n".join(f"{name}=" for name in prompt.variables)
        self.fills.delete("1.0", "end")
        if prefill:
            self.fills.insert("1.0", prefill)

    # -- values ---------------------------------------------------------------------
    def _fill_values(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for line in self.fills.get("1.0", "end").splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                if key.strip():
                    values[key.strip()] = value.strip()
        return values

    def _rendered(self) -> str:
        body = self.body.get("1.0", "end").rstrip()
        from ..prompts import PromptTemplate
        probe = PromptTemplate(id="tmp", title="t", category="c", body=body)
        return probe.render(self._fill_values())

    # -- actions -----------------------------------------------------------------------
    def send_to_chat(self) -> None:
        text = self._rendered()
        if not text.strip():
            self.app.show_info("Prompts", "The prompt is empty.")
            return
        from . import chat_tab
        chat_tab.queue_outgoing(self.app, text)
        self.app.select_tab("Chat")
        self.app.set_status("Prompt loaded into Chat - press Send there.")
        log_event("prompt sent to chat")

    def copy_prompt(self) -> None:
        self.app.root.clipboard_clear()
        self.app.root.clipboard_append(self._rendered())
        self.app.set_status("Prompt copied to the clipboard.")

    def save_as_mine(self) -> None:
        try:
            prompt = self.library.add(
                self.title_var.get() or "My prompt",
                self.edit_category_var.get() or "My prompts",
                self.body.get("1.0", "end").rstrip(),
            )
        except Exception as exc:
            self.app.show_error(exc)
            return
        self.library.save()
        self._reload_list()
        self.current_id = prompt.id
        self.app.set_status(f"Saved '{prompt.title}' to your prompts.")
        log_event(f"prompt added: {prompt.title!r}")

    def save_changes(self) -> None:
        if not self.current_id:
            self.app.show_info("Prompts", "Pick a prompt first, or use 'Save as mine'.")
            return
        try:
            prompt = self.library.update(
                self.current_id, self.title_var.get(),
                self.edit_category_var.get(), self.body.get("1.0", "end").rstrip(),
            )
        except Exception as exc:
            self.app.show_error(exc)
            return
        self.library.save()
        self._reload_list()
        self.app.set_status(f"Updated '{prompt.title}'.")

    def delete_current(self) -> None:
        if not self.current_id:
            return
        try:
            prompt = self.library.get(self.current_id)
        except Exception as exc:
            self.app.show_error(exc)
            return
        verb = "Hide" if prompt.builtin else "Delete"
        if not self.app.ask_yes_no(verb + " prompt", f"{verb} '{prompt.title}'?"):
            return
        self.library.delete(prompt.id)
        self.library.save()
        self.current_id = None
        self._reload_list()
        self.app.set_status(f"{verb} done for '{prompt.title}'.")

    def restore(self) -> None:
        self.library.restore_builtins()
        self.library.save()
        self._reload_list()
        self.app.set_status("Built-in prompts restored.")


def build(notebook, app) -> ttk.Frame:
    frame = ttk.Frame(notebook)
    PromptsTab(app, frame)
    return frame
