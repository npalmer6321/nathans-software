"""Best-effort, standard-library-only system probing.

Everything here degrades gracefully: a fact we cannot determine comes back
as ``None`` (displayed as "unknown") rather than raising. No third-party
packages, no shelling out to platform tools.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .ollama_client import human_bytes


@dataclass
class DiskInfo:
    path: str
    total_bytes: int
    free_bytes: int

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.free_bytes

    @property
    def free_fraction(self) -> float:
        return self.free_bytes / self.total_bytes if self.total_bytes else 0.0


@dataclass
class SystemSnapshot:
    os_name: str
    os_version: str
    architecture: str
    python_version: str
    cpu_count: int | None
    total_ram_bytes: int | None
    disks: list[DiskInfo]
    ollama_version_hint: str = ""

    @property
    def total_ram_human(self) -> str:
        return human_bytes(self.total_ram_bytes) if self.total_ram_bytes else "unknown"

    def ram_meets(self, minimum_gb: float) -> bool | None:
        """True/False whether RAM meets a floor, or None if unknown."""
        if not self.total_ram_bytes:
            return None
        return self.total_ram_bytes >= minimum_gb * 1024 ** 3


def _ram_windows() -> int | None:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:
        return None
    return None


def _ram_linux() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    return int(parts[1]) * 1024  # /proc reports kB
    except (OSError, ValueError, IndexError):
        return None
    return None


def _ram_macos() -> int | None:
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return int(out.stdout.strip())
    except (OSError, ValueError, ImportError):
        return None
    return None


def total_ram_bytes() -> int | None:
    """Total physical RAM in bytes, or None when undeterminable."""
    if sys.platform.startswith("win"):
        return _ram_windows()
    if sys.platform == "darwin":
        return _ram_macos()
    return _ram_linux()


def disk_info(path: str | os.PathLike[str]) -> DiskInfo | None:
    try:
        usage = shutil.disk_usage(str(path))
    except OSError:
        return None
    return DiskInfo(path=str(path), total_bytes=usage.total, free_bytes=usage.free)


def interesting_disks() -> list[DiskInfo]:
    """Home directory disk, plus the system drive on Windows."""
    candidates: list[Path] = [Path.home()]
    if sys.platform.startswith("win"):
        system_drive = os.environ.get("SystemDrive", "C:")
        candidates.append(Path(system_drive + "\\"))
    seen: list[DiskInfo] = []
    seen_paths: set[str] = set()
    for candidate in candidates:
        info = disk_info(candidate)
        if info and info.path not in seen_paths:
            seen.append(info)
            seen_paths.add(info.path)
    return seen


def snapshot(extra_disk: str | os.PathLike[str] | None = None) -> SystemSnapshot:
    """Collect the full dashboard snapshot."""
    disks = interesting_disks()
    if extra_disk:
        info = disk_info(extra_disk)
        if info and all(d.path != info.path for d in disks):
            disks.append(info)
    return SystemSnapshot(
        os_name=platform.system() or sys.platform,
        os_version=platform.version(),
        architecture=platform.machine() or "unknown",
        python_version=platform.python_version(),
        cpu_count=os.cpu_count(),
        total_ram_bytes=total_ram_bytes(),
        disks=disks,
    )


def format_snapshot(snap: SystemSnapshot) -> str:
    """Render a snapshot as plain text for the dashboard/log views."""
    lines = [
        f"OS: {snap.os_name} ({snap.architecture})",
        f"OS build: {snap.os_version}",
        f"Python: {snap.python_version}",
        f"CPU cores: {snap.cpu_count if snap.cpu_count else 'unknown'}",
        f"RAM: {snap.total_ram_human}",
    ]
    for disk in snap.disks:
        lines.append(
            f"Disk {disk.path}: {human_bytes(disk.free_bytes)} free of "
            f"{human_bytes(disk.total_bytes)} ({disk.free_fraction:.0%} free)"
        )
    return "\n".join(lines)


# Practical guidance numbers for the Qwen2.5-Coder family on CPU-only PCs.
MODEL_RAM_GUIDE: dict[str, tuple[float, str]] = {
    "qwen2.5-coder:7b": (8.0, "Comfortable on 16 GB - snappy for autocomplete-size tasks."),
    "qwen2.5-coder:14b": (24.0, "Runs on 16 GB, tight; closing other apps helps. 32 GB is comfy."),
    "qwen-coder-14b-16k": (24.0, "Same footprint as the 14B base, with the 16K context baked in."),
}


def ram_advice(model_name: str, snap: SystemSnapshot) -> str:
    """Plain-English RAM guidance for a model, given this machine."""
    guide = MODEL_RAM_GUIDE.get(model_name)
    if not guide:
        return "No specific RAM guidance for this model yet."
    recommended, note = guide
    meets = snap.ram_meets(recommended)
    if meets is None:
        return note
    if meets:
        return f"Your RAM meets the ~{recommended:.0f} GB comfort point. {note}"
    return f"Below the ~{recommended:.0f} GB comfort point - expect slower replies. {note}"
