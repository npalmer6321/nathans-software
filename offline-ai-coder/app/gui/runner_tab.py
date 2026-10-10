"""Run & Test tab: structured, button-triggered command execution.

There is deliberately no free-text command box. The buttons below map to
the allowlisted builders in :mod:`app.runner`; the exact argv is shown
before anything runs, output is captured, and a plain-English verdict is
parsed out of unittest results. Optional: turn a failing run into a Chat
question ("Ask the AI about this output").
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

from .. import runner as runner_mod
from ..logstore import log_event
from ..runner import RunResult
from ..themes import text_widget_kwargs


class RunnerTab:
    def __init__(self, app, frame: ttk.Frame) -> None:
        self.app = app
        self.frame = frame
        self.results: list[RunResult] = []
        self._build()

    def _build(self) -> None:
        app = self.app
        top = ttk.Frame(self.frame)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Project folder:", style="Dim.TLabel").pack(side="left")
        self.dir_var = tk.StringVar(value=app.current_project_path)
        ttk.Entry(top, textvariable=self.dir_var).pack(side="left", fill="x",
                                                      expand=True, padx=6)
        ttk.Button(top, text="Browse...", command=self.browse).pack(side="left")
        ttk.Button(top, text="Use Files-tab folder",
                   command=self.sync_from_files).pack(side="left", padx=4)

        actions = ttk.Labelframe(self.frame, text="What to run (nothing runs until you click)")
        actions.pack(fill="x", padx=8, pady=4)
        row = ttk.Frame(actions)
        row.pack(fill="x", padx=8, pady=6)
        ttk.Button(row, text="Run all unit tests", style="Accent.TButton",
                   command=self.run_all_tests).pack(side="left")
        ttk.Button(row, text="Run one test file...",
                   command=self.run_one_test).pack(side="left", padx=4)
        ttk.Button(row, text="Run a Python file...",
                   command=self.run_python_file).pack(side="left", padx=4)
        row2 = ttk.Frame(actions)
        row2.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Button(row2, text="Check Python version",
                   command=self.run_python_version).pack(side="left")
        ttk.Button(row2, text="List Ollama models (CLI)",
                   command=self.run_ollama_list).pack(side="left", padx=4)
        timeout = ttk.Frame(row2)
        timeout.pack(side="right")
        ttk.Label(timeout, text="Stop after (seconds):", style="Panel.TLabel").pack(side="left")
        self.timeout_var = tk.StringVar(value=str(int(app.settings.runner_timeout_seconds)))
        ttk.Entry(timeout, textvariable=self.timeout_var, width=6).pack(side="left", padx=4)

        self.command_var = tk.StringVar(value="The exact command will appear here before it runs.")
        ttk.Label(self.frame, textvariable=self.command_var,
                  style="Dim.TLabel").pack(anchor="w", padx=8, pady=(4, 0))
        self.verdict_var = tk.StringVar(value="")
        ttk.Label(self.frame, textvariable=self.verdict_var,
                  style="Dim.TLabel").pack(anchor="w", padx=8)
        self.output = tk.Text(self.frame, state="disabled",
                              **text_widget_kwargs(app.palette, code=True,
                                                   font_size=app.settings.font_size))
        self.output.pack(fill="both", expand=True, padx=8, pady=6)
        bottom = ttk.Frame(self.frame)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bottom, text="Ask the AI about this output",
                   command=self.ask_ai).pack(side="left")
        ttk.Button(bottom, text="Copy output", command=self.copy_output).pack(side="left", padx=4)
        ttk.Button(bottom, text="Clear", command=self.clear).pack(side="left", padx=4)
        self.history_var = tk.StringVar(value="Run history: nothing yet.")
        ttk.Label(bottom, textvariable=self.history_var, style="Dim.TLabel").pack(side="right")

    # -- helpers ---------------------------------------------------------------------
    def browse(self) -> None:
        chosen = filedialog.askdirectory(title="Choose a project folder",
                                         initialdir=self.dir_var.get() or None)
        if chosen:
            self.dir_var.set(chosen)

    def sync_from_files(self) -> None:
        if self.app.current_project_path:
            self.dir_var.set(self.app.current_project_path)
            self.app.set_status("Using the folder from the Files tab.")

    def _timeout(self) -> float:
        try:
            value = float(self.timeout_var.get())
        except ValueError:
            value = self.app.settings.runner_timeout_seconds
        return max(5.0, min(value, 3600.0))

    def _set_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")
        self.output.see("end")

    def _execute(self, spec) -> None:
        app = self.app
        self.command_var.set("Command: " + spec.display())
        if not app.ask_yes_no("Run this command?",
                              "Run exactly this?\n\n" + spec.display()):
            app.set_status("Run cancelled.")
            return
        self._set_output("Running...\n")
        self.verdict_var.set("")
        timeout = self._timeout()
        log_event(f"runner: {spec.display()} (timeout {timeout:.0f}s)")

        def work():
            return runner_mod.run_spec(spec, timeout=timeout)

        def done(result: RunResult) -> None:
            self.results.append(result)
            text = result.combined_output or "(no output)"
            self._set_output(text)
            verdict = result.summary_line()
            if "unittest" in " ".join(result.argv):
                parsed = runner_mod.parse_unittest_output(result.combined_output)
                verdict += " - " + parsed.render()
            self.verdict_var.set("Result: " + verdict)
            self.history_var.set("Run history: " + " | ".join(
                r.summary_line() for r in self.results[-4:]))
            app.set_status("Run finished: " + result.summary_line())
            log_event(f"runner result: {result.summary_line()}")

        def failed(exc) -> None:
            self._set_output(f"Could not run: {exc}")
            app.set_status("Run failed to start.")
            app.show_error(exc)

        app.run_async(work, done, failed)

    # -- the allowlisted actions ------------------------------------------------------
    def run_all_tests(self) -> None:
        try:
            spec = runner_mod.build_run_unittest(self.dir_var.get())
        except Exception as exc:
            self.app.show_error(exc)
            return
        self._execute(spec)

    def run_one_test(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Choose a test file", initialdir=self.dir_var.get() or None,
            filetypes=[("Python files", "*.py")])
        if not chosen:
            return
        try:
            spec = runner_mod.build_run_single_test(chosen, self.dir_var.get() or ".")
        except Exception as exc:
            self.app.show_error(exc)
            return
        self._execute(spec)

    def run_python_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Choose a Python file to run", initialdir=self.dir_var.get() or None,
            filetypes=[("Python files", "*.py")])
        if not chosen:
            return
        try:
            spec = runner_mod.build_run_python_file(chosen, self.dir_var.get() or ".")
        except Exception as exc:
            self.app.show_error(exc)
            return
        self._execute(spec)

    def run_python_version(self) -> None:
        self._execute(runner_mod.build_python_version())

    def run_ollama_list(self) -> None:
        try:
            spec = runner_mod.build_ollama_list()
        except Exception as exc:
            self.app.show_error(exc)
            return
        self._execute(spec)

    # -- extras ---------------------------------------------------------------------------
    def ask_ai(self) -> None:
        text = self.output.get("1.0", "end").strip()
        if not text or text == "Running...":
            self.app.show_info("Run & Test", "Run something first, then ask about its output.")
            return
        prompt = ("Here is the output from running my code/tests. Explain what "
                  "happened in plain English, and if something failed, show me "
                  "the fix.\n\n```\n" + text[-6000:] + "\n```")
        from . import chat_tab
        chat_tab.queue_outgoing(self.app, prompt)
        self.app.select_tab("Chat")

    def copy_output(self) -> None:
        self.app.root.clipboard_clear()
        self.app.root.clipboard_append(self.output.get("1.0", "end"))
        self.app.set_status("Output copied.")

    def clear(self) -> None:
        self._set_output("")
        self.verdict_var.set("")


def build(notebook, app) -> ttk.Frame:
    frame = ttk.Frame(notebook)
    RunnerTab(app, frame)
    return frame


def refresh(app) -> None:
    pass
