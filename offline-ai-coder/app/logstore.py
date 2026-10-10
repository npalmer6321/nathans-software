"""Application log handling: a small rotating log plus read/search helpers.

The app writes one line per notable event (server checks, model actions,
command runs, errors) to ``app.log`` in the data directory, rotating at
~1 MB with three backups. The Logs tab reads through this module, so all
file handling stays testable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import user_data_dir

LOG_FILENAME = "app.log"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3

_logger: logging.Logger | None = None


def log_path() -> Path:
    return user_data_dir() / LOG_FILENAME


def get_logger() -> logging.Logger:
    """The app-wide logger, configured on first use."""
    global _logger
    if _logger is not None:
        return _logger
    logger = logging.getLogger("offline_ai_coder")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        handler = RotatingFileHandler(
            log_path(), maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    except OSError:
        logger.addHandler(logging.NullHandler())
    _logger = logger
    return logger


def log_event(message: str, level: int = logging.INFO) -> None:
    """Write an event, never raising (logging must not break the app)."""
    try:
        get_logger().log(level, message)
    except Exception:
        pass


def log_error(message: str, exc: BaseException | None = None) -> None:
    text = f"{message}: {exc}" if exc is not None else message
    log_event(text, logging.ERROR)


@dataclass
class LogLine:
    raw: str
    timestamp: str = ""
    level: str = ""
    message: str = ""

    @property
    def is_error(self) -> bool:
        return self.level in ("ERROR", "CRITICAL")


_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(?P<level>[A-Z]+)\] (?P<msg>.*)$"
)


def parse_line(raw: str) -> LogLine:
    match = _LINE_RE.match(raw.rstrip("\n"))
    if not match:
        return LogLine(raw=raw.rstrip("\n"), message=raw.rstrip("\n"))
    return LogLine(
        raw=raw.rstrip("\n"),
        timestamp=match.group("ts"),
        level=match.group("level"),
        message=match.group("msg"),
    )


def read_log(path: Path | None = None, max_lines: int = 1000) -> list[LogLine]:
    """Read the newest ``max_lines`` of the log (oldest first)."""
    target = path or log_path()
    if not target.exists():
        return []
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = [parse_line(line) for line in text.splitlines() if line.strip()]
    return lines[-max_lines:]


def search_log(lines: list[LogLine], needle: str, errors_only: bool = False) -> list[LogLine]:
    """Filter parsed lines by substring and/or errors-only."""
    result = []
    lowered = needle.strip().lower()
    for line in lines:
        if errors_only and not line.is_error:
            continue
        if lowered and lowered not in line.raw.lower():
            continue
        result.append(line)
    return result


def render_lines(lines: list[LogLine]) -> str:
    if not lines:
        return "The log is empty."
    return "\n".join(line.raw for line in lines)


def recent_errors(path: Path | None = None, limit: int = 5) -> list[LogLine]:
    lines = read_log(path)
    errors = [line for line in lines if line.is_error]
    return errors[-limit:]


def stamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
