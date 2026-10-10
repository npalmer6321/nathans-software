"""A small, dependency-free client for the local Ollama HTTP API.

Only the Python standard library is used (``urllib``). Every failure mode
maps onto a typed error from :mod:`app.errors` so the GUI can distinguish
"server is not running" (start Ollama) from "server said no" (fix the
request). All network calls accept a timeout and never retry silently.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

from .errors import OllamaHTTPError, OllamaProtocolError, OllamaUnreachable

ProgressFn = Callable[[str, float | None], None]
"""Called with (human message, optional 0..1 fraction) during long calls."""

DEFAULT_TIMEOUT = 30.0
PULL_TIMEOUT = 3600.0  # model downloads can legitimately take a long time


@dataclass
class ModelInfo:
    """One installed model, as reported by ``GET /api/tags``."""

    name: str
    size_bytes: int = 0
    modified_at: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""

    @property
    def size_human(self) -> str:
        return human_bytes(self.size_bytes)

    @property
    def base_name(self) -> str:
        return self.name.split(":")[0]

    @property
    def tag(self) -> str:
        return self.name.split(":", 1)[1] if ":" in self.name else "latest"


@dataclass
class ServerStatus:
    """Snapshot of server reachability."""

    reachable: bool
    version: str = ""
    model_count: int = 0
    detail: str = ""

    @property
    def summary(self) -> str:
        if not self.reachable:
            return "Ollama is not answering"
        return f"Ollama {self.version or '?'} - {self.model_count} model(s)"


@dataclass
class ChatMessage:
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class ChatResult:
    content: str
    model: str = ""
    total_duration_ns: int = 0
    eval_count: int = 0

    @property
    def tokens_per_second(self) -> float | None:
        if not self.total_duration_ns or not self.eval_count:
            return None
        seconds = self.total_duration_ns / 1_000_000_000
        return self.eval_count / seconds if seconds > 0 else None


def human_bytes(num: int | float) -> str:
    """Format a byte count the way humans expect (1024-based)."""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"  # pragma: no cover


def parse_tags_payload(payload: object) -> list[ModelInfo]:
    """Turn a decoded ``/api/tags`` body into :class:`ModelInfo` rows."""
    if not isinstance(payload, dict):
        raise OllamaProtocolError("Model list was not a JSON object.")
    models = payload.get("models")
    if models is None:
        return []
    if not isinstance(models, list):
        raise OllamaProtocolError("Model list 'models' field was not a list.")
    parsed: list[ModelInfo] = []
    for entry in models:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        details = entry.get("details") or {}
        parsed.append(
            ModelInfo(
                name=str(entry["name"]),
                size_bytes=int(entry.get("size") or 0),
                modified_at=str(entry.get("modified_at") or ""),
                family=str(details.get("family") or ""),
                parameter_size=str(details.get("parameter_size") or ""),
                quantization=str(details.get("quantization_level") or ""),
            )
        )
    parsed.sort(key=lambda m: m.name.lower())
    return parsed


def parse_pull_line(line: str) -> tuple[str, float | None]:
    """Parse one streamed pull-progress line into (message, fraction)."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return (line.strip(), None)
    if not isinstance(data, dict):
        return (line.strip(), None)
    status = str(data.get("status") or "working")
    total = data.get("total")
    completed = data.get("completed")
    fraction: float | None = None
    if isinstance(total, (int, float)) and total and isinstance(completed, (int, float)):
        fraction = max(0.0, min(1.0, completed / total))
        status = f"{status} - {human_bytes(completed)} / {human_bytes(total)}"
    return (status, fraction)


def parse_chat_payload(payload: object, model: str = "") -> ChatResult:
    """Turn a decoded ``/api/chat`` (or ``/api/generate``) body into a result."""
    if not isinstance(payload, dict):
        raise OllamaProtocolError("Chat response was not a JSON object.")
    message = payload.get("message")
    content = ""
    if isinstance(message, dict):
        content = str(message.get("content") or "")
    elif isinstance(payload.get("response"), str):
        content = payload["response"]
    return ChatResult(
        content=content,
        model=str(payload.get("model") or model),
        total_duration_ns=int(payload.get("total_duration") or 0),
        eval_count=int(payload.get("eval_count") or 0),
    )


def iter_json_lines(byte_stream: Iterable[bytes]) -> Iterator[dict]:
    """Yield decoded JSON objects from a stream of newline-delimited JSON."""
    for raw in byte_stream:
        text = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OllamaProtocolError(f"Unparsable line from Ollama: {text[:120]!r}") from exc
        if isinstance(obj, dict):
            yield obj


class OllamaClient:
    """Synchronous Ollama client bound to one base URL."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- transport -------------------------------------------------------
    def _request(self, method: str, path: str, payload: dict | None = None,
                 timeout: float | None = None) -> bytes:
        url = self.base_url + path
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise OllamaHTTPError(exc.code, body) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OllamaUnreachable(
                f"Could not reach Ollama at {self.base_url}. Is it running? ({exc})"
            ) from exc

    def _request_stream(self, path: str, payload: dict, timeout: float) -> Iterator[dict]:
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise OllamaHTTPError(exc.code, body) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OllamaUnreachable(
                f"Could not reach Ollama at {self.base_url}. Is it running? ({exc})"
            ) from exc
        with response:
            yield from iter_json_lines(response)

    def _get_json(self, path: str, timeout: float | None = None) -> object:
        raw = self._request("GET", path, timeout=timeout)
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise OllamaProtocolError(f"Ollama answered {path} with non-JSON data.") from exc

    # -- public API ------------------------------------------------------
    def ping(self) -> bool:
        """True if the server answers at all."""
        try:
            self._request("GET", "/", timeout=min(self.timeout, 5.0))
            return True
        except OllamaUnreachable:
            return False

    def version(self) -> str:
        payload = self._get_json("/api/version")
        if isinstance(payload, dict):
            return str(payload.get("version") or "")
        return ""

    def list_models(self) -> list[ModelInfo]:
        payload = self._get_json("/api/tags")
        return parse_tags_payload(payload)

    def status(self) -> ServerStatus:
        """One-shot health snapshot used by the dashboard."""
        try:
            version = self.version()
            models = self.list_models()
        except OllamaUnreachable as exc:
            return ServerStatus(False, detail=str(exc))
        except OllamaHTTPError as exc:
            return ServerStatus(False, detail=str(exc))
        return ServerStatus(True, version=version, model_count=len(models))

    def show_model(self, name: str) -> dict:
        raw = self._request("POST", "/api/show", {"name": name})
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise OllamaProtocolError("Model details were not JSON.") from exc
        return data if isinstance(data, dict) else {}

    def delete_model(self, name: str) -> None:
        self._request("DELETE", "/api/delete", {"name": name})

    def pull_model(self, name: str, progress: ProgressFn | None = None,
                   timeout: float = PULL_TIMEOUT) -> str:
        """Download a model, reporting progress. Returns the last status line."""
        last = "starting"
        for obj in self._request_stream("/api/pull", {"name": name, "stream": True}, timeout):
            if "error" in obj:
                raise OllamaHTTPError(500, str(obj["error"]))
            text, fraction = parse_pull_line(json.dumps(obj))
            last = text
            if progress:
                progress(text, fraction)
        if progress:
            progress("done", 1.0)
        return last

    def chat(self, model: str, messages: list[ChatMessage], temperature: float = 0.2,
             num_ctx: int | None = None, timeout: float | None = None) -> ChatResult:
        """Single non-streaming chat completion."""
        payload: dict = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if num_ctx:
            payload["options"]["num_ctx"] = num_ctx
        raw = self._request("POST", "/api/chat", payload, timeout=timeout or self.timeout)
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise OllamaProtocolError("Chat response was not JSON.") from exc
        return parse_chat_payload(data, model)

    def chat_stream(self, model: str, messages: list[ChatMessage], temperature: float = 0.2,
                    num_ctx: int | None = None, timeout: float | None = None) -> Iterator[str]:
        """Yield reply text chunks as they arrive."""
        payload: dict = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {"temperature": temperature},
        }
        if num_ctx:
            payload["options"]["num_ctx"] = num_ctx
        for obj in self._request_stream("/api/chat", payload, timeout or self.timeout):
            if "error" in obj:
                raise OllamaHTTPError(500, str(obj["error"]))
            message = obj.get("message")
            if isinstance(message, dict) and message.get("content"):
                yield str(message["content"])

    def generate(self, model: str, prompt: str, system: str = "", temperature: float = 0.2,
                 timeout: float | None = None) -> str:
        """Single-shot text completion via ``/api/generate``."""
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        raw = self._request("POST", "/api/generate", payload, timeout=timeout or self.timeout)
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise OllamaProtocolError("Generate response was not JSON.") from exc
        return parse_chat_payload(data, model).content
