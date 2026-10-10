"""Safe command runner.

Design rule (enforced here, not just in the GUI): this module never runs
free-text shell commands. The GUI offers a short menu of structured
actions - "run this Python file", "run the tests in this folder" - and
each action is turned into an argv list by code in this module, after
validating every piece. Processes run without a shell, with a timeout,
captured output, and a hard cap on output size.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .errors import CommandNotAllowed, CommandTimedOut, RunnerError

MAX_OUTPUT_CHARS = 200_000
DEFAULT_TIMEOUT = 120.0

# Program names that may be resolved from PATH (never user-typed paths
# containing separators - those must come from structured pickers).
ALLOWED_TOOLS = {"python", "python3", "pytest", "ollama"}


@dataclass
class CommandSpec:
    """A validated command, ready to run."""

    argv: list[str]
    cwd: Path
    label: str

    def display(self) -> str:
        return " ".join(f'"{a}"' if " " in a else a for a in self.argv)


@dataclass
class RunResult:
    argv: list[str]
    cwd: str
    returncode: int | None
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def combined_output(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(self.stderr)
        return "\n".join(parts).strip()

    def summary_line(self) -> str:
        if self.timed_out:
            return f"Timed out after {self.seconds:.1f}s"
        state = "succeeded" if self.ok else f"failed (exit {self.returncode})"
        return f"{state} in {self.seconds:.1f}s"

    def argv_command(self) -> str:
        return " ".join(self.argv)


@dataclass
class TestSummary:
    ran: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    ok: bool | None = None  # None = could not tell
    raw_tail: str = ""

    def render(self) -> str:
        if self.ok is None:
            return "Could not tell how the tests went - see the full output."
        if self.ok:
            return f"All {self.ran} test(s) passed."
        bits = [f"{self.ran} ran", f"{self.failures} failed", f"{self.errors} errored"]
        if self.skipped:
            bits.append(f"{self.skipped} skipped")
        return "Tests: " + ", ".join(bits) + "."


# -- command builders (the allowlist) --------------------------------------

def _require_dir(path: str | Path) -> Path:
    folder = Path(path).expanduser()
    if not folder.is_dir():
        raise CommandNotAllowed(f"Not a folder: {folder}")
    return folder


def _require_file(path: str | Path, suffixes: set[str] | None = None) -> Path:
    target = Path(path).expanduser()
    if not target.is_file():
        raise CommandNotAllowed(f"Not a file: {target}")
    if suffixes and target.suffix.lower() not in suffixes:
        raise CommandNotAllowed(
            f"Only these file types can be run directly: {', '.join(sorted(suffixes))}"
        )
    return target


def build_run_python_file(file_path: str | Path, project_dir: str | Path) -> CommandSpec:
    target = _require_file(file_path, {".py"})
    cwd = _require_dir(project_dir)
    return CommandSpec([sys.executable, str(target)], cwd, f"Run {target.name}")


def build_run_unittest(project_dir: str | Path, pattern: str = "test*.py") -> CommandSpec:
    cwd = _require_dir(project_dir)
    if not re.fullmatch(r"[A-Za-z0-9_.*-]+\.py", pattern):
        raise CommandNotAllowed(f"Unsafe test pattern: {pattern!r}")
    argv = [sys.executable, "-m", "unittest", "discover", "-s", str(cwd),
            "-p", pattern, "-v"]
    return CommandSpec(argv, cwd, "Run unit tests (unittest)")


def build_run_single_test(file_path: str | Path, project_dir: str | Path) -> CommandSpec:
    target = _require_file(file_path, {".py"})
    cwd = _require_dir(project_dir)
    argv = [sys.executable, "-m", "unittest", str(target), "-v"]
    return CommandSpec(argv, cwd, f"Run tests in {target.name}")


def build_python_version() -> CommandSpec:
    return CommandSpec([sys.executable, "--version"], Path.cwd(), "Python version")


def build_ollama_list() -> CommandSpec:
    return CommandSpec(["ollama", "list"], Path.cwd(), "Ollama model list")


def validate_spec(spec: CommandSpec) -> None:
    """Final gate before anything runs."""
    if not spec.argv:
        raise CommandNotAllowed("Empty command.")
    program = Path(spec.argv[0]).name.lower()
    program_stem = program[:-4] if program.endswith(".exe") else program
    if spec.argv[0] != sys.executable and program_stem not in ALLOWED_TOOLS:
        raise CommandNotAllowed(f"Program is not on the allowlist: {spec.argv[0]}")
    for arg in spec.argv[1:]:
        if any(ch in arg for ch in ("\n", "\r")):
            raise CommandNotAllowed("Arguments may not contain newlines.")
    if not spec.cwd.is_dir():
        raise CommandNotAllowed(f"Working folder is missing: {spec.cwd}")


# -- execution ---------------------------------------------------------------

def _cap_output(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    return text[:MAX_OUTPUT_CHARS] + "\n... [output truncated] ...", True


def run_spec(spec: CommandSpec, timeout: float = DEFAULT_TIMEOUT) -> RunResult:
    """Run a validated spec to completion (or timeout)."""
    validate_spec(spec)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            spec.argv, cwd=str(spec.cwd), capture_output=True, text=True,
            timeout=timeout, shell=False,
        )
        seconds = time.monotonic() - started
        stdout, trunc_out = _cap_output(proc.stdout or "")
        stderr, trunc_err = _cap_output(proc.stderr or "")
        return RunResult(spec.argv, str(spec.cwd), proc.returncode, stdout,
                         stderr, seconds, truncated=trunc_out or trunc_err)
    except subprocess.TimeoutExpired as exc:
        seconds = time.monotonic() - started
        raise CommandTimedOut(timeout, partial_output=_as_text(exc.stdout)) from exc
    except FileNotFoundError as exc:
        raise RunnerError(f"Program not found: {spec.argv[0]}") from exc
    except OSError as exc:
        raise RunnerError(f"Could not start the command: {exc}") from exc


def _as_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


# -- test output parsing -----------------------------------------------------

_RAN_RE = re.compile(r"Ran (\d+) test")
_FAIL_RE = re.compile(r"failures=(\d+)")
_ERR_RE = re.compile(r"errors=(\d+)")
_SKIP_RE = re.compile(r"skipped=(\d+)")


def parse_unittest_output(output: str) -> TestSummary:
    """Extract counts from `python -m unittest` output (stdout or stderr)."""
    summary = TestSummary(raw_tail="\n".join(output.splitlines()[-15:]))
    ran = _RAN_RE.search(output)
    if not ran:
        return summary
    summary.ran = int(ran.group(1))
    fail = _FAIL_RE.search(output)
    err = _ERR_RE.search(output)
    skip = _SKIP_RE.search(output)
    summary.failures = int(fail.group(1)) if fail else 0
    summary.errors = int(err.group(1)) if err else 0
    summary.skipped = int(skip.group(1)) if skip else 0
    if "\nOK" in output or output.strip().endswith("OK"):
        summary.ok = True
    elif "FAILED" in output:
        summary.ok = False
    else:
        summary.ok = summary.failures == 0 and summary.errors == 0
    return summary


def recent_results_text(results: list[RunResult]) -> str:
    if not results:
        return "Nothing has been run yet this session."
    return "\n".join(
        f"- {r.argv_command()} -> {r.summary_line()}" for r in results
    )
