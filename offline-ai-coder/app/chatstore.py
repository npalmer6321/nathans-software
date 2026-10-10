"""Chat session model and persistence.

A chat is an ordered list of messages with a title, the model in use, and
timestamps. Sessions are kept in memory for the Chat tab and can be saved
to / loaded from ``chats.json`` so a conversation survives restarts. The
model itself never sees more than a capped window of history (see
:meth:`ChatSession.window`).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import user_data_dir

CHATS_FILENAME = "chats.json"
MAX_STORED_SESSIONS = 25
MAX_TITLE_CHARS = 60


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str
    when: float = field(default_factory=time.time)

    def as_payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatSession:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "New chat"
    model: str = ""
    messages: list[Message] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    # -- conversation handling --------------------------------------------
    def add(self, role: str, content: str) -> Message:
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.updated = time.time()
        if role == "user" and self.title == "New chat":
            first = " ".join(content.split())
            self.title = (first[:MAX_TITLE_CHARS] + "...") if len(first) > MAX_TITLE_CHARS else first
        return message

    def clear(self) -> None:
        keep = [m for m in self.messages if m.role == "system"]
        self.messages = keep
        self.title = "New chat"
        self.updated = time.time()

    def window(self, max_messages: int = 30) -> list[Message]:
        """System message (if any) plus the most recent turns."""
        system = [m for m in self.messages if m.role == "system"][:1]
        rest = [m for m in self.messages if m.role != "system"]
        return system + rest[-max_messages:]

    def payload(self, max_messages: int = 30) -> list[dict[str, str]]:
        return [m.as_payload() for m in self.window(max_messages)]

    def transcript(self) -> str:
        """Human-readable transcript for copying out of the app."""
        lines = [f"# {self.title}", f"Model: {self.model or 'unknown'}", ""]
        labels = {"system": "System", "user": "You", "assistant": "AI"}
        for message in self.messages:
            lines.append(f"**{labels.get(message.role, message.role)}:**")
            lines.append(message.content)
            lines.append("")
        return "\n".join(lines).strip()

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")


def session_to_dict(session: ChatSession) -> dict:
    return asdict(session)


def session_from_dict(data: object) -> ChatSession | None:
    if not isinstance(data, dict):
        return None
    try:
        messages = [
            Message(role=str(m["role"]), content=str(m["content"]),
                    when=float(m.get("when", time.time())))
            for m in (data.get("messages") or [])
            if isinstance(m, dict) and "role" in m and "content" in m
        ]
        return ChatSession(
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            title=str(data.get("title") or "New chat"),
            model=str(data.get("model") or ""),
            messages=messages,
            created=float(data.get("created", time.time())),
            updated=float(data.get("updated", time.time())),
        )
    except (TypeError, ValueError):
        return None


def save_sessions(sessions: list[ChatSession], path: Path | None = None) -> Path:
    target = path or (user_data_dir() / CHATS_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    trimmed = sorted(sessions, key=lambda s: s.updated, reverse=True)[:MAX_STORED_SESSIONS]
    payload = {"sessions": [session_to_dict(s) for s in trimmed]}
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def load_sessions(path: Path | None = None) -> list[ChatSession]:
    target = path or (user_data_dir() / CHATS_FILENAME)
    if not target.exists():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    found: list[ChatSession] = []
    for entry in (data.get("sessions") if isinstance(data, dict) else None) or []:
        session = session_from_dict(entry)
        if session is not None:
            found.append(session)
    return sorted(found, key=lambda s: s.updated, reverse=True)


DEFAULT_SYSTEM_PROMPT = (
    "You are a practical coding assistant running fully offline on the "
    "user's own PC. Give complete, working code with short explanations. "
    "The user is learning to program, so prefer clarity over cleverness, "
    "explain new terms briefly, and never assume internet access."
)


def new_session(model: str = "", system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> ChatSession:
    session = ChatSession(model=model)
    if system_prompt:
        session.messages.append(Message(role="system", content=system_prompt))
    return session
