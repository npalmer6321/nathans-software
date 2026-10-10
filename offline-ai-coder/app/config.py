"""Persistent application settings.

Settings live in a per-user JSON file:

* Windows: ``%LOCALAPPDATA%\\OfflineCodingAgent\\settings.json``
* macOS:   ``~/Library/Application Support/OfflineCodingAgent/settings.json``
* Linux:   ``$XDG_CONFIG_HOME/offline-coding-agent/settings.json`` (or ``~/.config``)

The module is deliberately forgiving: a missing file yields defaults, a
corrupt file yields defaults *plus* a recorded warning (the bad file is
kept as ``settings.json.bak`` so nothing is silently destroyed), and
unknown keys from newer/older versions are preserved on save.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .errors import ConfigError

APP_DIR_NAME = "OfflineCodingAgent"
SETTINGS_FILENAME = "settings.json"

# Models the companion installer pulls; used as sensible defaults.
DEFAULT_MAIN_MODEL = "qwen-coder-14b-16k"
DEFAULT_FAST_MODEL = "qwen2.5-coder:7b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"


def user_data_dir() -> Path:
    """Return (creating if needed) the per-user data directory."""
    override = os.environ.get("OFFLINE_AI_CODER_HOME")
    if override:
        path = Path(override).expanduser()
    elif sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        path = Path(base) / APP_DIR_NAME
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        path = Path(base) / "offline-coding-agent"
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path() -> Path:
    return user_data_dir() / SETTINGS_FILENAME


@dataclass
class Settings:
    """Every user-tweakable setting, with safe defaults."""

    ollama_base_url: str = DEFAULT_BASE_URL
    main_model: str = DEFAULT_MAIN_MODEL
    fast_model: str = DEFAULT_FAST_MODEL
    default_model_role: str = "main"  # "main" or "fast"
    request_timeout_seconds: float = 30.0
    chat_timeout_seconds: float = 300.0
    temperature: float = 0.2
    num_ctx: int = 16384
    theme: str = "dark"  # "dark" or "light"
    font_size: int = 10
    last_project_dir: str = ""
    recent_projects: list[str] = field(default_factory=list)
    max_recent_projects: int = 8
    runner_timeout_seconds: float = 120.0
    max_file_preview_bytes: int = 512_000
    confirm_before_delete_model: bool = True
    check_server_on_start: bool = True

    # Keys we do not model explicitly but must round-trip untouched.
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- derived helpers -------------------------------------------------
    @property
    def active_model(self) -> str:
        return self.fast_model if self.default_model_role == "fast" else self.main_model

    def remember_project(self, path: str | os.PathLike[str]) -> None:
        """Move ``path`` to the front of the recent-projects list."""
        text = str(Path(path))
        if text in self.recent_projects:
            self.recent_projects.remove(text)
        self.recent_projects.insert(0, text)
        del self.recent_projects[self.max_recent_projects :]
        self.last_project_dir = text

    # -- validation ------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty means fine)."""
        problems: list[str] = []
        if not self.ollama_base_url.startswith(("http://", "https://")):
            problems.append("Ollama address must start with http:// or https://")
        if self.request_timeout_seconds <= 0:
            problems.append("Request timeout must be greater than zero")
        if self.chat_timeout_seconds < self.request_timeout_seconds:
            problems.append("Chat timeout should be at least the request timeout")
        if not 0.0 <= self.temperature <= 2.0:
            problems.append("Temperature must be between 0.0 and 2.0")
        if self.num_ctx < 512:
            problems.append("Context size must be at least 512 tokens")
        if self.default_model_role not in ("main", "fast"):
            problems.append("Default model role must be 'main' or 'fast'")
        if self.theme not in ("dark", "light"):
            problems.append("Theme must be 'dark' or 'light'")
        if not 8 <= self.font_size <= 20:
            problems.append("Font size must be between 8 and 20")
        if self.max_recent_projects < 1:
            problems.append("Recent project limit must be at least 1")
        if self.runner_timeout_seconds < 5:
            problems.append("Runner timeout must be at least 5 seconds")
        if self.max_file_preview_bytes < 1024:
            problems.append("File preview limit must be at least 1 KB")
        return problems


_KNOWN_FIELDS = {f.name for f in fields(Settings)} - {"extra"}


def settings_from_dict(data: dict[str, Any]) -> Settings:
    """Build a :class:`Settings` from parsed JSON, tolerating junk."""
    if not isinstance(data, dict):
        raise ConfigError("Settings file did not contain a JSON object.")
    known = {key: value for key, value in data.items() if key in _KNOWN_FIELDS}
    extra = {key: value for key, value in data.items() if key not in _KNOWN_FIELDS}
    try:
        settings = Settings(**known)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Settings file had invalid values: {exc}") from exc
    settings.extra = extra
    return settings


def settings_to_dict(settings: Settings) -> dict[str, Any]:
    data = asdict(settings)
    extra = data.pop("extra", {}) or {}
    merged = dict(extra)
    merged.update(data)
    return merged


@dataclass
class LoadResult:
    settings: Settings
    warning: str | None = None
    created_new: bool = False


def load_settings(path: Path | None = None) -> LoadResult:
    """Load settings from disk, falling back to defaults when needed."""
    target = path or settings_path()
    if not target.exists():
        return LoadResult(Settings(), created_new=True)
    try:
        raw = target.read_text(encoding="utf-8")
        data = json.loads(raw)
        return LoadResult(settings_from_dict(data))
    except (OSError, json.JSONDecodeError, ConfigError) as exc:
        backup = target.with_suffix(".json.bak")
        try:
            shutil.copy2(target, backup)
            note = f" Kept a copy at {backup}."
        except OSError:
            note = ""
        warning = f"Could not read settings ({exc}); using defaults.{note}"
        return LoadResult(Settings(), warning=warning)


def save_settings(settings: Settings, path: Path | None = None) -> Path:
    """Persist settings atomically (write temp file, then replace)."""
    problems = settings.validate()
    if problems:
        raise ConfigError("Settings are not valid: " + "; ".join(problems))
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    payload = json.dumps(settings_to_dict(settings), indent=2, sort_keys=False)
    try:
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        raise ConfigError(f"Could not save settings to {target}: {exc}") from exc
    return target


def reset_settings(path: Path | None = None) -> Settings:
    """Save and return factory-default settings."""
    settings = Settings()
    save_settings(settings, path)
    return settings


def describe_environment() -> dict[str, str]:
    """Small facts shown on the dashboard footer."""
    return {
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "settings_file": str(settings_path()),
        "data_dir": str(user_data_dir()),
    }
