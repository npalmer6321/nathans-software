"""Files tab: browse a project folder, preview files, feed context to Chat.

Also owns "current project" state shared with the Chat and Run & Test
tabs: opening a folder here updates ``app.current_project_path`` and the
settings' recent-projects list.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

from .. import projects as projects_mod
from ..logstore import log_event
from ..projects import FileNode, Project
from ..themes import text_widget_kwargs


class FilesTab:
    def __init__(self, app, frame: ttk.Frame) -> None:
        self.app = app
        self.frame = frame
        self.project: Project | None = None
        self._build()
        last = app.settings.last_project_dir
        if last:
            self.path_var.set(last)

    def _build(self) -> None:
        app = self.app
        top = ttk.Frame(self.frame)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Project folder:", style="Dim.TLabel").pack(side="left")
        self.path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.path_var).pack(side="left", fill="x",
                                                       expand=True, padx=6)
        ttk.Button(top, text="Browse...", command=self.browse).pack(side="left")
        ttk.Button(top, text="Open", style="Accent.TButton",
                   command=self.open_project).pack(side="left", padx=4)
        recent = ttk.Combobox(top, values=app.settings.recent_projects, width=22)
        recent.pack(side="left", padx=4)
        recent.bind("<<ComboboxSelected>>",
                    lambda e: (self.path_var.set(recent.get()), self.open_project()))
        self.recent_box = recent

        self.summary_var = tk.StringVar(value="No project open. The bundled sample "
                                              "project works too - try the 'sample' folder.")
        ttk.Label(self.frame, textvariable=self.summary_var,
                  style="Dim.TLabel").pack(anchor="w", padx=8)

        body = ttk.Frame(self.frame)
        body.pack(fill="both", expand=True, padx=8, pady=6)
        self.tree = ttk.Treeview(body, show="tree", height=22)
        self.tree.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.preview = tk.Text(body, state="disabled",
                               **text_widget_kwargs(app.palette, code=True,
                                                    font_size=app.settings.font_size))
        self.preview.pack(side="left", fill="both", expand=True, padx=(6, 0))

        bottom = ttk.Frame(self.frame)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        self.file_info = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.file_info, style="Dim.TLabel").pack(side="left")
        ttk.Button(bottom, text="Send file to Chat input",
                   command=self.send_to_chat).pack(side="right")
        ttk.Button(bottom, text="Use this folder in Chat",
                   command=self.use_in_chat).pack(side="right", padx=4)
        self._selected_path = ""

    # -- project handling ----------------------------------------------------
    def browse(self) -> None:
        chosen = filedialog.askdirectory(title="Choose a project folder",
                                         initialdir=self.app.settings.last_project_dir or None)
        if chosen:
            self.path_var.set(chosen)
            self.open_project()

    def open_project(self) -> None:
        path = self.path_var.get().strip()
        if not path:
            self.app.show_info("Files", "Pick a project folder first.")
            return
        app = self.app

        def work():
            return projects_mod.scan_project(path)

        def done(project: Project) -> None:
            self.project = project
            app.current_project_path = str(project.root)
            app.settings.remember_project(project.root)
            app.save_settings_safely()
            self.recent_box["values"] = app.settings.recent_projects
            self._fill_tree(project.tree)
            self.summary_var.set(
                f"{project.name}: {project.summary_text()} "
                f"(size on disk includes only counted files)")
            app.set_status(f"Project open: {project.name}")
            log_event(f"project opened: {project.root} ({project.file_count} files)")

        app.run_async(work, done, app.show_error)

    def _fill_tree(self, root: FileNode) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        def add(node: FileNode, parent: str) -> None:
            for child in node.children:
                label = child.name + ("/" if child.is_dir else "")
                item = self.tree.insert(parent, "end", text=label,
                                        values=(str(child.path),))
                if child.is_dir:
                    add(child, item)

        add(root, "")

    # -- selection/preview ------------------------------------------------------
    def _on_select(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        values = self.tree.item(selection[0], "values")
        if not values:
            return
        path = values[0]
        self._selected_path = path
        from pathlib import Path
        if Path(path).is_dir():
            self.file_info.set(f"Folder: {path}")
            return
        app = self.app
        limit = app.settings.max_file_preview_bytes

        def work():
            return projects_mod.read_preview(path, limit)

        def done(preview) -> None:
            if preview.binary:
                text = f"[binary file - {preview.size_bytes} bytes - not shown]"
            else:
                text = preview.text
                if preview.truncated:
                    text += "\n\n... [truncated for display] ..."
            self._set_preview(text)
            self.file_info.set(
                f"{preview.path.name} - {preview.size_bytes} bytes"
                + (" (truncated)" if preview.truncated else ""))

        app.run_async(work, done, app.show_error)

    def _set_preview(self, text: str) -> None:
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    # -- hand-offs ------------------------------------------------------------------
    def send_to_chat(self) -> None:
        if not self._selected_path:
            self.app.show_info("Files", "Pick a file in the tree first.")
            return
        try:
            preview = projects_mod.read_preview(
                self._selected_path, self.app.settings.max_file_preview_bytes)
        except Exception as exc:
            self.app.show_error(exc)
            return
        if preview.binary:
            self.app.show_info("Files", "That file is binary - the model cannot read it as text.")
            return
        from pathlib import Path
        text = f"Here is my file {Path(self._selected_path).name}:\n\n```\n{preview.text}\n```\n\n"
        from . import chat_tab
        chat_tab.queue_outgoing(self.app, text)
        self.app.select_tab("Chat")
        self.app.set_status("File loaded into Chat - add your question and press Send.")

    def use_in_chat(self) -> None:
        if self.project is None:
            self.app.show_info("Files", "Open a project folder first.")
            return
        self.app.select_tab("Chat")
        self.app.set_status(
            "Chat will include this project's files when 'Include my project files' is on.")


def build(notebook, app) -> ttk.Frame:
    frame = ttk.Frame(notebook)
    FilesTab(app, frame)
    return frame


def refresh(app) -> None:
    pass
