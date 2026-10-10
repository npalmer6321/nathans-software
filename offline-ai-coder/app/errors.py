"""Typed errors shared across the Offline AI Coder app.

Keeping error types small and explicit lets the GUI show friendly,
actionable messages instead of raw tracebacks, and lets the core logic be
tested without a display.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application-level errors."""


class ConfigError(AppError):
    """Raised when settings cannot be read, parsed, or saved."""


class OllamaError(AppError):
    """Base class for Ollama communication problems."""


class OllamaUnreachable(OllamaError):
    """The Ollama server did not answer at all (offline, not installed, down)."""


class OllamaHTTPError(OllamaError):
    """The Ollama server answered with a non-2xx HTTP status."""

    def __init__(self, status: int, body: str = "") -> None:
        self.status = status
        self.body = body
        message = f"Ollama returned HTTP {status}"
        if body:
            message += f": {body.strip()[:300]}"
        super().__init__(message)


class OllamaProtocolError(OllamaError):
    """The Ollama server answered with something we could not parse."""


class RunnerError(AppError):
    """Base class for command-running problems."""


class CommandNotAllowed(RunnerError):
    """The requested command is not on the safety allowlist."""


class CommandTimedOut(RunnerError):
    """A command exceeded its timeout and was stopped."""

    def __init__(self, timeout: float, partial_output: str = "") -> None:
        self.timeout = timeout
        self.partial_output = partial_output
        super().__init__(f"Command timed out after {timeout:.0f} seconds and was stopped.")


class ProjectError(AppError):
    """Raised for file/project browsing problems (missing paths, too big, etc.)."""
