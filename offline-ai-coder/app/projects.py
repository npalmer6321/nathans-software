"""Project folder browsing model.

Walks a project directory into a tree the GUI can display, skipping the
usual noise (``.git``, virtualenvs, caches, build output), guards against
absurd file sizes, and detects binary files so previews never splatter
control characters across the screen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ProjectError

IGNORE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", "env", "node_modules", ".idea", ".vscode", "build",
    "dist", "target", ".gradle", ".tox",
}
IGNORE_SUFFIXES = {".pyc", ".pyo", ".class", ".o", ".obj", ".dll", ".exe",
                   ".so", ".dylib", ".bin"}
TEXT_EXTENSIONS = {
    ".py", ".txt", ".md", ".rst", ".json", ".toml", ".yaml", ".yml", ".ini",
    ".cfg", ".csv", ".kt", ".kts", ".java", ".js", ".ts", ".html", ".css",
    ".c", ".h", ".cpp", ".hpp", ".rs", ".go", ".sh", ".bat", ".ps1",
    ".xml", ".gradle", ".properties", ".gitignore",
}
BINARY_SNIFF_BYTES = 4096


@dataclass
class FileNode:
    name: str
    path: Path
    is_dir: bool
    size_bytes: int = 0
    children: list["FileNode"] = field(default_factory=list)

    @property
    def depth_key(self) -> tuple:
        return (0 if self.is_dir else 1, self.name.lower())

    def count_files(self) -> int:
        if not self.is_dir:
            return 1
        return sum(child.count_files() for child in self.children)

    def total_size(self) -> int:
        if not self.is_dir:
            return self.size_bytes
        return sum(child.total_size() for child in self.children)

    def find(self, relative: str) -> "FileNode | None":
        parts = Path(relative).parts
        node: FileNode | None = self
        for part in parts:
            if node is None or not node.is_dir:
                return None
            node = next((c for c in node.children if c.name == part), None)
        return node

    def flatten(self, prefix: str = "") -> list[tuple[str, "FileNode"]]:
        """(relative path, node) pairs for every node, depth-first."""
        rows: list[tuple[str, FileNode]] = []
        for child in sorted(self.children, key=lambda c: c.depth_key):
            rel = f"{prefix}{child.name}"
            rows.append((rel + ("/" if child.is_dir else ""), child))
            if child.is_dir:
                rows.extend(child.flatten(rel + "/"))
        return rows


@dataclass
class Project:
    root: Path
    tree: FileNode

    @property
    def name(self) -> str:
        return self.root.name or str(self.root)

    @property
    def file_count(self) -> int:
        return self.tree.count_files()

    def language_summary(self) -> dict[str, int]:
        """Extension -> file count, most common first."""
        counts: dict[str, int] = {}
        for _, node in self.tree.flatten():
            if not node.is_dir:
                ext = node.path.suffix.lower() or "(no extension)"
                counts[ext] = counts.get(ext, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary_text(self) -> str:
        langs = self.language_summary()
        top = ", ".join(f"{ext} x{count}" for ext, count in list(langs.items())[:5])
        return f"{self.file_count} files - {top}" if top else "empty project"


def scan_project(root: str | os.PathLike[str], max_depth: int = 12,
                 max_entries: int = 5000) -> Project:
    """Walk ``root`` into a :class:`Project` tree."""
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise ProjectError(f"Folder does not exist: {root_path}")
    if not root_path.is_dir():
        raise ProjectError(f"Not a folder: {root_path}")
    counter = [0]
    tree = _scan_dir(root_path, depth=0, max_depth=max_depth,
                     max_entries=max_entries, counter=counter)
    return Project(root=root_path, tree=tree)


def _scan_dir(path: Path, depth: int, max_depth: int, max_entries: int,
              counter: list[int]) -> FileNode:
    node = FileNode(name=path.name, path=path, is_dir=True)
    if depth >= max_depth:
        return node
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name.lower())
    except OSError as exc:
        raise ProjectError(f"Cannot read folder {path}: {exc}") from exc
    for entry in entries:
        if counter[0] >= max_entries:
            break
        name = entry.name
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if is_dir:
            if name in IGNORE_DIRS or (name.startswith(".") and name not in (".", "..")):
                continue
            counter[0] += 1
            node.children.append(
                _scan_dir(Path(entry.path), depth + 1, max_depth, max_entries, counter)
            )
        else:
            if Path(name).suffix.lower() in IGNORE_SUFFIXES:
                continue
            counter[0] += 1
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                size = 0
            node.children.append(FileNode(name=name, path=Path(entry.path),
                                          is_dir=False, size_bytes=size))
    node.children.sort(key=lambda c: c.depth_key)
    return node


def looks_binary(sample: bytes) -> bool:
    """Heuristic: a NUL byte in the first chunk means binary."""
    return b"\x00" in sample[:BINARY_SNIFF_BYTES]


def is_likely_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as handle:
            return not looks_binary(handle.read(BINARY_SNIFF_BYTES))
    except OSError:
        return False


@dataclass
class FilePreview:
    path: Path
    text: str
    truncated: bool
    binary: bool
    size_bytes: int


def read_preview(path: str | os.PathLike[str], max_bytes: int = 512_000) -> FilePreview:
    """Read up to ``max_bytes`` of a text file for display."""
    target = Path(path).expanduser()
    if not target.exists():
        raise ProjectError(f"File does not exist: {target}")
    if target.is_dir():
        raise ProjectError(f"That is a folder, not a file: {target}")
    size = target.stat().st_size
    with open(target, "rb") as handle:
        raw = handle.read(max_bytes + 1)
    truncated = len(raw) > max_bytes
    raw = raw[:max_bytes]
    if looks_binary(raw):
        return FilePreview(target, "", truncated, True, size)
    text = raw.decode("utf-8", errors="replace")
    return FilePreview(target, text, truncated, False, size)


def find_test_files(project: Project) -> list[Path]:
    """Heuristically locate test files (for the runner's quick actions)."""
    found: list[Path] = []
    for _, node in project.tree.flatten():
        if node.is_dir:
            continue
        name = node.name.lower()
        if name.startswith("test_") or name.endswith("_test.py") or "tests" in node.path.parts:
            if name.endswith(".py"):
                found.append(node.path)
    return found


def project_context(project: Project, max_files: int = 12,
                    max_chars_per_file: int = 4000) -> str:
    """A compact 'here is my project' brief to paste into a chat prompt.

    Small source files are included (truncated) so the model can actually
    answer questions about the code; big and binary files are listed only.
    """
    lines = [f"Project: {project.name} ({project.summary_text()})", ""]
    listing = project.tree.flatten()
    lines.append("Files:")
    for rel, node in listing[:200]:
        if not node.is_dir:
            lines.append(f"  {rel}")
    included = 0
    for _, node in listing:
        if included >= max_files:
            break
        if node.is_dir or not is_likely_text_file(node.path):
            continue
        if node.size_bytes > 60_000:
            continue
        preview = read_preview(node.path, max_chars_per_file)
        if preview.binary:
            continue
        included += 1
        lines.append("")
        lines.append(f"--- {node.path.name} ---")
        lines.append(preview.text)
    return "\n".join(lines)
