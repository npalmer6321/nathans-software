"""Chat tab: talk to the local coder model, with saveable sessions.

Replies stream in token-by-token from a worker thread into a thread-safe
queue; the UI drains it with ``after`` so typing never stutters. The
"include my project" toggle prepends a compact project brief (from the
Files tab's folder) so the model can answer about real code.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

from .. import projects as projects_mod
from ..chatstore import ChatSession, load_sessions, new_session, save_sessions
from ..errors import OllamaError
from ..logstore import log_error, log_event
from ..ollama_client import ChatMessage
from ..themes import text_widget_kwargs


class ChatTab:
    def __init__(self, app, frame: ttk.Frame) -> None:
        self.app = app
        self.frame = frame
        self.sessions: list[ChatSession] = load_sessions()
        self.session: ChatSession = (
            self.sessions[0] if self.sessions else new_session(app.settings.active_model)
        )
        if self.session.model != app.settings.active_model:
            self.session.model = app.settings.active_model
        self._chunks: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._busy = False
        self._build()
        self._render_transcript()
        frame.after(60, self._drain_chunks)

    # -- layout ---------------------------------------------------------------
    def _build(self) -> None:
        app = self.app
        top = ttk.Frame(self.frame)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Model:", style="Dim.TLabel").pack(side="left")
        self.model_var = tk.StringVar(value=self.session.model)
        self.model_box = ttk.Combobox(top, textvariable=self.model_var, width=28)
        self.model_box.pack(side="left", padx=6)
        ttk.Button(top, text="Load models", command=self._load_models).pack(side="left")
        self.include_project = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Include my project files",
                        variable=self.include_project).pack(side="left", padx=10)
        ttk.Button(top, text="New chat", command=self.new_chat).pack(side="right")
        ttk.Button(top, text="Copy chat", command=self.copy_chat).pack(side="right", padx=4)
        ttk.Button(top, text="Save chat", command=self.save_chat).pack(side="right")

        second = ttk.Frame(self.frame)
        second.pack(fill="x", padx=8)
        ttk.Label(second, text="Saved chats:", style="Dim.TLabel").pack(side="left")
        self.saved_var = tk.StringVar()
        self.saved_box = ttk.Combobox(second, textvariable=self.saved_var,
                                      state="readonly", width=42)
        self.saved_box.pack(side="left", padx=6)
        ttk.Button(second, text="Open", command=self.open_saved).pack(side="left")
        ttk.Button(second, text="Delete", command=self.delete_saved).pack(side="left", padx=4)
        self._refresh_saved_list()

        self.transcript = tk.Text(self.frame, state="disabled",
                                  **text_widget_kwargs(app.palette,
                                                       font_size=app.settings.font_size))
        self.transcript.pack(fill="both", expand=True, padx=8, pady=6)
        self.transcript.tag_configure("you", foreground=app.palette.accent,
                                      font=("Segoe UI", app.settings.font_size, "bold"))
        self.transcript.tag_configure("ai", foreground=app.palette.good,
                                      font=("Segoe UI", app.settings.font_size, "bold"))
        self.transcript.tag_configure("sys", foreground=app.palette.fg_dim)

        bottom = ttk.Frame(self.frame)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        self.input = tk.Text(bottom, height=4,
                             **text_widget_kwargs(app.palette,
                                                  font_size=app.settings.font_size))
        self.input.pack(side="left", fill="x", expand=True)
        self.input.bind("<Control-Return>", lambda _e: self.send())
        side = ttk.Frame(bottom)
        side.pack(side="left", padx=(6, 0))
        self.send_button = ttk.Button(side, text="Send  (Ctrl+Enter)",
                                      style="Accent.TButton", command=self.send)
        self.send_button.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready. The model runs 100% on this PC.")
        ttk.Label(side, textvariable=self.status_var, style="Dim.TLabel",
                  wraplength=180).pack(pady=(6, 0))

    # -- transcript rendering ---------------------------------------------------
    def _render_transcript(self) -> None:
        widget = self.transcript
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        for message in self.session.messages:
            if message.role == "system":
                continue
            tag = "you" if message.role == "user" else "ai"
            label = "You" if message.role == "user" else "AI"
            widget.insert("end", f"{label}:  ", tag)
            widget.insert("end", message.content + "\n\n")
        widget.configure(state="disabled")
        widget.see("end")

    def _append(self, label: str, tag: str, text: str) -> None:
        widget = self.transcript
        widget.configure(state="normal")
        if label:
            widget.insert("end", label, tag)
        widget.insert("end", text)
        widget.configure(state="disabled")
        widget.see("end")

    def _refresh_saved_list(self) -> None:
        titles = [f"{s.title}  ({s.turn_count} q)" for s in self.sessions]
        self.saved_box["values"] = titles
        if titles:
            self.saved_var.set(titles[0])

    # -- session actions ---------------------------------------------------------
    def new_chat(self) -> None:
        self.session = new_session(self.app.settings.active_model)
        self.model_var.set(self.session.model)
        self._render_transcript()
        self.status_var.set("New chat started.")
        log_event("chat: new session")

    def save_chat(self) -> None:
        if self.session.turn_count == 0:
            self.app.show_info("Chat", "Nothing to save yet - ask something first.")
            return
        existing = {s.id: i for i, s in enumerate(self.sessions)}
        if self.session.id in existing:
            self.sessions[existing[self.session.id]] = self.session
        else:
            self.sessions.insert(0, self.session)
        save_sessions(self.sessions)
        self._refresh_saved_list()
        self.status_var.set(f"Saved '{self.session.title}'.")
        log_event(f"chat saved: {self.session.title!r} ({self.session.turn_count} turns)")

    def open_saved(self) -> None:
        index = self.saved_box.current()
        if index < 0 or index >= len(self.sessions):
            self.app.show_info("Chat", "Pick a saved chat first.")
            return
        self.session = self.sessions[index]
        self.model_var.set(self.session.model)
        self._render_transcript()
        self.status_var.set(f"Opened '{self.session.title}'.")

    def delete_saved(self) -> None:
        index = self.saved_box.current()
        if index < 0 or index >= len(self.sessions):
            return
        victim = self.sessions.pop(index)
        save_sessions(self.sessions)
        self._refresh_saved_list()
        self.status_var.set(f"Deleted '{victim.title}'.")
        log_event(f"chat deleted: {victim.title!r}")

    def copy_chat(self) -> None:
        text = self.session.transcript()
        self.app.root.clipboard_clear()
        self.app.root.clipboard_append(text)
        self.status_var.set("Chat copied to the clipboard.")

    def on_close(self, app) -> None:  # called by main window hook signature (module-level)
        pass

    # -- model list ---------------------------------------------------------------
    def _load_models(self) -> None:
        app = self.app

        def work():
            return [m.name for m in app.client.list_models()]

        def done(names: list[str]) -> None:
            self.model_box["values"] = names
            if names and self.model_var.get() not in names:
                self.model_var.set(names[0])
            self.status_var.set(f"{len(names)} model(s) available.")

        app.run_async(work, done, app.show_error)

    # -- sending ---------------------------------------------------------------------
    def _project_context(self) -> str:
        path = self.app.current_project_path
        if not path:
            return ""
        try:
            project = projects_mod.scan_project(path)
            return projects_mod.project_context(project)
        except Exception as exc:
            log_error("project context failed", exc)
            return ""

    def send(self) -> None:
        if self._busy:
            return
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        app = self.app
        model = self.model_var.get().strip() or app.settings.active_model
        self.session.model = model

        outgoing = text
        if self.include_project.get():
            context = self._project_context()
            if context:
                outgoing = ("Here is context about my project:\n\n" + context +
                            "\n\nMy question: " + text)

        self.session.add("user", text)
        self.input.delete("1.0", "end")
        self._append("You:  ", "you", text + "\n\n")
        self._append("AI:  ", "ai", "")
        self._busy = True
        self.send_button.configure(state="disabled")
        self.status_var.set(f"Thinking with {model} (local)...")

        history = [ChatMessage(m["role"], m["content"])
                   for m in self.session.payload()]
        # Swap the last user entry for the context-augmented copy if needed.
        if self.include_project.get() and history:
            history[-1] = ChatMessage("user", outgoing)

        settings = app.settings

        def work() -> None:
            try:
                for chunk in app.client.chat_stream(
                    model, history, temperature=settings.temperature,
                    num_ctx=settings.num_ctx, timeout=settings.chat_timeout_seconds,
                ):
                    self._chunks.put(("chunk", chunk))
                self._chunks.put(("done", ""))
            except OllamaError as exc:
                self._chunks.put(("error", str(exc)))
            except Exception as exc:  # surfaced to the user, never swallowed
                self._chunks.put(("error", f"Unexpected error: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def _drain_chunks(self) -> None:
        reply_parts: list[str] = getattr(self, "_reply_parts", [])
        try:
            while True:
                kind, payload = self._chunks.get_nowait()
                if kind == "chunk":
                    reply_parts.append(payload)
                    self._append("", "", payload)
                elif kind == "done":
                    self._finish_reply("".join(reply_parts))
                    reply_parts = []
                elif kind == "error":
                    self._finish_reply("".join(reply_parts), error=payload)
                    reply_parts = []
        except Exception:
            pass
        self._reply_parts = reply_parts
        if self.frame.winfo_exists():
            self.frame.after(60, self._drain_chunks)

    def _finish_reply(self, full_text: str, error: str = "") -> None:
        self._busy = False
        self.send_button.configure(state="normal")
        if error:
            self._append("", "", f"\n[Stopped: {error}]\n\n")
            self.status_var.set("The model could not answer - is Ollama running?")
            self.session.add("assistant", f"[error: {error}]")
            log_error("chat failed", OllamaError(error))
        else:
            if full_text:
                self.session.add("assistant", full_text)
            self._append("", "", "\n\n")
            self.status_var.set("Done. Everything stayed on this PC.")
        self.frame.update_idletasks()


_tab: ChatTab | None = None


def build(notebook, app) -> ttk.Frame:
    global _tab
    frame = ttk.Frame(notebook)
    _tab = ChatTab(app, frame)
    return frame


def queue_outgoing(app, text: str) -> None:
    """Drop prepared text (e.g. a filled prompt) into the input box."""
    if _tab is None:
        return
    _tab.input.delete("1.0", "end")
    _tab.input.insert("1.0", text)
    _tab.input.focus_set()


def on_close(app) -> None:
    """Persist the active session if it has content worth keeping."""
    if _tab is not None and _tab.session.turn_count:
        try:
            existing = {s.id for s in _tab.sessions}
            if _tab.session.id not in existing:
                _tab.sessions.insert(0, _tab.session)
            save_sessions(_tab.sessions)
        except Exception as exc:
            log_error("could not save chats on exit", exc)
