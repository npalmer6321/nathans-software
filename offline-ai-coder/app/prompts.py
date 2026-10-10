"""Prompt library for coding help.

Ships with a set of practical, copy-ready prompt templates for a local
coder model, and lets the user add/edit/delete their own. Templates may
contain ``{variables}`` that get filled in at use time (for example
``{language}``, ``{code}``, ``{error}``). Library state is stored as JSON
in the app's data directory; built-ins are always present and can be
hidden but not destroyed (a reset restores them).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .config import user_data_dir
from .errors import AppError

PROMPTS_FILENAME = "prompts.json"
_VARIABLE_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PromptError(AppError):
    """Prompt library problems (bad template, unknown id, ...)."""


@dataclass
class PromptTemplate:
    id: str
    title: str
    category: str
    body: str
    builtin: bool = False
    hidden: bool = False

    @property
    def variables(self) -> list[str]:
        """Variable names the template needs, in first-appearance order."""
        seen: list[str] = []
        for name in _VARIABLE_RE.findall(self.body):
            if name not in seen:
                seen.append(name)
        return seen

    def render(self, values: dict[str, str] | None = None) -> str:
        """Substitute variables; missing ones are left as ``{name}``."""
        values = values or {}

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            return values.get(key, match.group(0))

        return _VARIABLE_RE.sub(replace, self.body)

    def render_strict(self, values: dict[str, str]) -> str:
        """Like :meth:`render` but raises when a variable has no value."""
        missing = [name for name in self.variables if name not in values]
        if missing:
            raise PromptError(
                "Missing value(s) for: " + ", ".join("{" + m + "}" for m in missing)
            )
        return self.render(values)


def _builtin(title: str, category: str, body: str) -> PromptTemplate:
    return PromptTemplate(
        id="builtin-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"),
        title=title, category=category, body=body.strip(), builtin=True,
    )


BUILTIN_PROMPTS: list[PromptTemplate] = [
    _builtin("Explain this code", "Understanding", """
Explain the following {language} code to me in plain English. Cover: what it
does overall, the important pieces line by line, and anything that looks
like a bug or a bad habit. Keep it practical - I am learning.

```{language}
{code}
```
"""),
    _builtin("Fix this error", "Debugging", """
I am getting this error in my {language} program. Explain what it means in
simple terms, point at the most likely cause in my code, and show me the
fixed code.

Error:
```
{error}
```

My code:
```{language}
{code}
```
"""),
    _builtin("Write a function", "Writing code", """
Write a {language} function that does this: {task}

Requirements:
- Include a short comment at the top explaining what it does
- Handle obvious edge cases (empty input, wrong types) sensibly
- Show one small example of calling it at the end
"""),
    _builtin("Add tests", "Testing", """
Write unit tests for the following {language} code using {framework}. Cover
the normal cases, the edge cases (empty input, None/null), and at least one
case that should raise an error. Explain in one sentence what each test
checks.

```{language}
{code}
```
"""),
    _builtin("Review my code", "Review", """
Review this {language} code like a friendly senior developer. List:
1. Bugs or things that will break
2. Things that are hard to read or maintain
3. Simple improvements, with the improved version of just those parts

```{language}
{code}
```
"""),
    _builtin("Explain like I'm new", "Understanding", """
Explain {concept} to me like I am smart but brand new to programming. Use a
real-world analogy, then a tiny {language} example (under 15 lines), then
tell me the one mistake beginners always make with it.
"""),
    _builtin("Turn steps into code", "Writing code", """
Turn these plain-English steps into a working {language} program. Keep it
simple and readable, comment each step, and tell me how to run it.

Steps:
{task}
"""),
    _builtin("Make it faster", "Review", """
This {language} code works but is slow on big inputs. Explain why in plain
English, then show a faster version and say what actually changed.

```{language}
{code}
```
"""),
    _builtin("Comment my code", "Writing code", """
Add clear comments to this {language} code without changing what it does.
Comment the purpose of each function and any tricky lines. Do not comment
the obvious.

```{language}
{code}
```
"""),
    _builtin("Plan a small project", "Planning", """
I want to build this: {task}

Give me a beginner-friendly plan: the smallest useful version first, the
files I will need, and the order to build them in. Keep the whole plan
achievable in {language} with no paid services.
"""),
]


@dataclass
class PromptLibrary:
    """Built-ins plus the user's own templates, with persistence."""

    user_prompts: list[PromptTemplate] = field(default_factory=list)
    hidden_builtin_ids: set[str] = field(default_factory=set)

    # -- queries ---------------------------------------------------------
    def all(self, include_hidden: bool = False) -> list[PromptTemplate]:
        visible = [
            p for p in BUILTIN_PROMPTS
            if include_hidden or p.id not in self.hidden_builtin_ids
        ]
        return visible + list(self.user_prompts)

    def categories(self) -> list[str]:
        cats: list[str] = []
        for prompt in self.all():
            if prompt.category not in cats:
                cats.append(prompt.category)
        return cats

    def by_category(self, category: str) -> list[PromptTemplate]:
        return [p for p in self.all() if p.category == category]

    def search(self, text: str) -> list[PromptTemplate]:
        needle = text.strip().lower()
        if not needle:
            return self.all()
        return [
            p for p in self.all()
            if needle in p.title.lower() or needle in p.body.lower()
            or needle in p.category.lower()
        ]

    def get(self, prompt_id: str) -> PromptTemplate:
        for prompt in self.all(include_hidden=True):
            if prompt.id == prompt_id:
                return prompt
        raise PromptError(f"No prompt with id {prompt_id!r}")

    # -- mutation --------------------------------------------------------
    def add(self, title: str, category: str, body: str) -> PromptTemplate:
        title, category, body = title.strip(), category.strip(), body.strip()
        if not title:
            raise PromptError("A prompt needs a title.")
        if not body:
            raise PromptError("A prompt needs some text.")
        prompt = PromptTemplate(
            id="user-" + uuid.uuid4().hex[:12],
            title=title, category=category or "My prompts", body=body,
        )
        self.user_prompts.append(prompt)
        return prompt

    def update(self, prompt_id: str, title: str, category: str, body: str) -> PromptTemplate:
        prompt = self.get(prompt_id)
        if prompt.builtin:
            raise PromptError("Built-in prompts cannot be edited; add a copy instead.")
        prompt.title = title.strip() or prompt.title
        prompt.category = category.strip() or prompt.category
        prompt.body = body.strip() or prompt.body
        return prompt

    def delete(self, prompt_id: str) -> None:
        prompt = self.get(prompt_id)
        if prompt.builtin:
            # Built-ins are hidden, not destroyed, so a reset is trivial.
            self.hidden_builtin_ids.add(prompt.id)
        else:
            self.user_prompts = [p for p in self.user_prompts if p.id != prompt.id]

    def restore_builtins(self) -> None:
        self.hidden_builtin_ids.clear()

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "user_prompts": [asdict(p) for p in self.user_prompts],
            "hidden_builtin_ids": sorted(self.hidden_builtin_ids),
        }

    @classmethod
    def from_dict(cls, data: object) -> "PromptLibrary":
        if not isinstance(data, dict):
            return cls()
        prompts: list[PromptTemplate] = []
        for entry in data.get("user_prompts") or []:
            if not isinstance(entry, dict):
                continue
            try:
                prompts.append(PromptTemplate(
                    id=str(entry["id"]), title=str(entry["title"]),
                    category=str(entry.get("category", "My prompts")),
                    body=str(entry["body"]), builtin=False,
                ))
            except (KeyError, TypeError):
                continue
        hidden = {str(x) for x in (data.get("hidden_builtin_ids") or [])}
        return cls(user_prompts=prompts, hidden_builtin_ids=hidden)

    def save(self, path: Path | None = None) -> Path:
        target = path or (user_data_dir() / PROMPTS_FILENAME)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "PromptLibrary":
        target = path or (user_data_dir() / PROMPTS_FILENAME)
        if not target.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return cls()


def variable_help(template: PromptTemplate) -> str:
    """One line describing what a template needs filled in."""
    variables = template.variables
    if not variables:
        return "No blanks to fill in - ready to send as-is."
    return "Fill in: " + ", ".join("{" + v + "}" for v in variables)


def expand_iter(templates: Iterable[PromptTemplate]) -> list[str]:
    return [t.title for t in templates]
