"""Setup checklist engine.

The companion installer (``Setup-OfflineAgent.ps1``) does the heavy lifting
once; this checklist answers "is my offline rig actually ready?" any time
afterwards. Each check is a small function returning a :class:`CheckItem`,
which keeps the logic testable without a real Ollama server: probes can be
injected, and the evaluation functions are pure given probe results.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from . import system_info
from .ollama_client import ModelInfo, OllamaClient, human_bytes

MIN_RECOMMENDED_FREE_GB = 12.0  # headroom for two Qwen2.5-Coder models
MIN_PYTHON = (3, 9)


class CheckStatus(Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"

    @property
    def symbol(self) -> str:
        return {"ok": "[OK]", "warn": "[!]", "fail": "[X]", "unknown": "[?]"}[self.value]


@dataclass
class CheckItem:
    key: str
    title: str
    status: CheckStatus
    detail: str
    advice: str = ""

    def render(self) -> str:
        text = f"{self.status.symbol} {self.title} - {self.detail}"
        if self.advice and self.status in (CheckStatus.WARN, CheckStatus.FAIL):
            text += f"\n    -> {self.advice}"
        return text


@dataclass
class ChecklistReport:
    items: list[CheckItem] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Fraction of checks that are OK (UNKNOWN counts as half)."""
        if not self.items:
            return 0.0
        points = {"ok": 1.0, "unknown": 0.5, "warn": 0.25, "fail": 0.0}
        return sum(points[i.status.value] for i in self.items) / len(self.items)

    @property
    def ready(self) -> bool:
        required = {"server", "main_model"}
        return all(
            item.status is CheckStatus.OK for item in self.items if item.key in required
        )

    def render(self) -> str:
        lines = [item.render() for item in self.items]
        pct = int(round(self.score * 100))
        lines.append("")
        lines.append(f"Readiness: {pct}% - " + ("ready to code offline." if self.ready
                                                else "a few steps remain (see [!] / [X] above)."))
        return "\n".join(lines)


# -- pure evaluators (probe results in, verdict out) -----------------------

def evaluate_python_version(version: tuple[int, int, int] = sys.version_info[:3]) -> CheckItem:
    pretty = ".".join(str(p) for p in version)
    if version >= MIN_PYTHON + (0,):
        return CheckItem("python", "Python version", CheckStatus.OK,
                         f"Python {pretty} found")
    return CheckItem("python", "Python version", CheckStatus.FAIL,
                     f"Python {pretty} is too old",
                     "Install Python 3.9 or newer from python.org and re-run setup.")


def evaluate_binary(name: str, found_path: str | None, friendly: str, advice: str) -> CheckItem:
    if found_path:
        return CheckItem(name, friendly, CheckStatus.OK, f"found at {found_path}")
    return CheckItem(name, friendly, CheckStatus.FAIL, "not found on PATH", advice)


def evaluate_server(reachable: bool, version: str = "", detail: str = "") -> CheckItem:
    if reachable:
        return CheckItem("server", "Ollama server answering", CheckStatus.OK,
                         f"responding (version {version or 'unknown'})")
    return CheckItem("server", "Ollama server answering", CheckStatus.FAIL,
                     detail or "no answer on the local address",
                     "Start Ollama (open the Ollama app, or run 'ollama serve'), then re-check.")


def evaluate_model(key: str, title: str, wanted: str, installed: list[ModelInfo],
                   required: bool) -> CheckItem:
    names = {m.name for m in installed}
    if wanted in names:
        match = next(m for m in installed if m.name == wanted)
        return CheckItem(key, title, CheckStatus.OK,
                         f"'{wanted}' installed ({match.size_human})")
    # A sibling tag of the same base still counts as present-but-different.
    base = wanted.split(":")[0]
    sibling = next((m for m in installed if m.base_name == base), None)
    if sibling is not None:
        return CheckItem(key, title, CheckStatus.WARN,
                         f"wanted '{wanted}', found '{sibling.name}' instead",
                         f"Pull the exact model with: ollama pull {wanted}")
    status = CheckStatus.FAIL if required else CheckStatus.WARN
    return CheckItem(key, title, status, f"'{wanted}' is not installed",
                     f"Download it once with: ollama pull {wanted}")


def evaluate_disk(free_bytes: int, total_bytes: int) -> CheckItem:
    free_gb = free_bytes / 1024 ** 3
    if free_gb >= MIN_RECOMMENDED_FREE_GB:
        return CheckItem("disk", "Free disk space", CheckStatus.OK,
                         f"{human_bytes(free_bytes)} free")
    return CheckItem("disk", "Free disk space", CheckStatus.WARN,
                     f"only {human_bytes(free_bytes)} free",
                     "Models are big; free up space before pulling new ones.")


def evaluate_ram(total_bytes: int | None) -> CheckItem:
    if not total_bytes:
        return CheckItem("ram", "System RAM", CheckStatus.UNKNOWN,
                         "could not be determined on this system")
    gb = total_bytes / 1024 ** 3
    if gb >= 30:
        return CheckItem("ram", "System RAM", CheckStatus.OK, f"{gb:.0f} GB - plenty")
    if gb >= 14:
        return CheckItem("ram", "System RAM", CheckStatus.OK,
                         f"{gb:.0f} GB - fine for the 7B, workable for the 14B")
    return CheckItem("ram", "System RAM", CheckStatus.WARN, f"{gb:.0f} GB - tight",
                     "Stick to the 7B model and close other apps while chatting.")


# -- live runner ------------------------------------------------------------

def run_checklist(client: OllamaClient, main_model: str, fast_model: str,
                  disk_path: str | Path | None = None) -> ChecklistReport:
    """Run every check against the real machine."""
    items: list[CheckItem] = []

    items.append(evaluate_python_version())
    items.append(evaluate_binary(
        "ollama_bin", shutil.which("ollama"), "Ollama program installed",
        "Run Setup-OfflineAgent.ps1 again, or install Ollama from ollama.com."))
    items.append(evaluate_binary(
        "aider", shutil.which("aider"), "Aider terminal coder",
        "Optional: pip install aider-chat, or use the Chat tab in this app."))

    status = client.status()
    installed: list[ModelInfo] = []
    if status.reachable:
        try:
            installed = client.list_models()
        except Exception:
            installed = []
    items.append(evaluate_server(status.reachable, status.version, status.detail))
    items.append(evaluate_model("main_model", "Main coder model", main_model,
                                installed, required=True))
    items.append(evaluate_model("fast_model", "Fast helper model", fast_model,
                                installed, required=False))

    snap = system_info.snapshot(disk_path)
    items.append(evaluate_ram(snap.total_ram_bytes))
    if snap.disks:
        first = snap.disks[0]
        items.append(evaluate_disk(first.free_bytes, first.total_bytes))
    else:
        items.append(CheckItem("disk", "Free disk space", CheckStatus.UNKNOWN,
                               "could not be measured"))
    return ChecklistReport(items)


REQUIRED_KEYS = ("server", "main_model")


def next_steps(report: ChecklistReport, limit: int = 3) -> list[str]:
    """The most important remaining actions, worst first."""
    order = {CheckStatus.FAIL: 0, CheckStatus.WARN: 1, CheckStatus.UNKNOWN: 2,
             CheckStatus.OK: 3}
    actionable = [i for i in report.items if i.status is not CheckStatus.OK and i.advice]
    actionable.sort(key=lambda i: order[i.status])
    return [f"{item.title}: {item.advice}" for item in actionable[:limit]]
