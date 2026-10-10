"""Color themes for the Tkinter GUI.

Two hand-tuned palettes (dark and light) applied through the ``clam``
ttk theme, which is the most cooperative built-in theme for custom
colors. Plain Tk widgets (Text, Listbox, menus) do not follow ttk styles,
so helpers here expose the raw palette for those widgets to use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str
    panel: str
    panel_alt: str
    fg: str
    fg_dim: str
    accent: str
    accent_fg: str
    good: str
    warn: str
    bad: str
    border: str
    code_bg: str


DARK = Palette(
    name="dark",
    bg="#14181d",
    panel="#1d232b",
    panel_alt="#242c37",
    fg="#e8edf2",
    fg_dim="#9fb0c3",
    accent="#3fa7ff",
    accent_fg="#06121f",
    good="#4cc38a",
    warn="#e5c07b",
    bad="#ff6b6b",
    border="#39434f",
    code_bg="#0d1117",
)

LIGHT = Palette(
    name="light",
    bg="#f4f6f8",
    panel="#ffffff",
    panel_alt="#e9eef3",
    fg="#1b2430",
    fg_dim="#5a6a7d",
    accent="#0b6bcb",
    accent_fg="#ffffff",
    good="#157347",
    warn="#9a6a00",
    bad="#c92a2a",
    border="#c3cedb",
    code_bg="#f6f8fa",
)

PALETTES = {"dark": DARK, "light": LIGHT}


def get_palette(theme_name: str) -> Palette:
    return PALETTES.get(theme_name, DARK)


def status_color(status_value: str, palette: Palette) -> str:
    """Map checklist/status words onto palette colors."""
    return {
        "ok": palette.good,
        "warn": palette.warn,
        "fail": palette.bad,
        "unknown": palette.fg_dim,
    }.get(status_value, palette.fg)


def apply_theme(root, palette: Palette, font_size: int = 10) -> None:
    """Apply ``palette`` to a Tk root window and its ttk widgets."""
    import tkinter as tk
    from tkinter import ttk

    root.configure(bg=palette.bg)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    base_font = ("Segoe UI", font_size) if palette else ("TkDefaultFont", font_size)
    mono_font = ("Consolas", font_size) if palette else ("TkFixedFont", font_size)

    style.configure(".", background=palette.bg, foreground=palette.fg,
                    font=base_font, bordercolor=palette.border)
    style.configure("TFrame", background=palette.bg)
    style.configure("Panel.TFrame", background=palette.panel)
    style.configure("TLabel", background=palette.bg, foreground=palette.fg)
    style.configure("Panel.TLabel", background=palette.panel, foreground=palette.fg)
    style.configure("Dim.TLabel", background=palette.bg, foreground=palette.fg_dim)
    style.configure("Title.TLabel", background=palette.bg, foreground=palette.fg,
                    font=("Segoe UI", font_size + 6, "bold"))
    style.configure("H2.TLabel", background=palette.bg, foreground=palette.fg,
                    font=("Segoe UI", font_size + 2, "bold"))
    style.configure("TButton", background=palette.panel_alt, foreground=palette.fg,
                    padding=(10, 5), borderwidth=1)
    style.map("TButton",
              background=[("active", palette.accent), ("disabled", palette.panel)],
              foreground=[("active", palette.accent_fg), ("disabled", palette.fg_dim)])
    style.configure("Accent.TButton", background=palette.accent,
                    foreground=palette.accent_fg, padding=(12, 6))
    style.map("Accent.TButton",
              background=[("active", palette.accent), ("disabled", palette.panel)],
              foreground=[("disabled", palette.fg_dim)])
    style.configure("TEntry", fieldbackground=palette.panel, foreground=palette.fg,
                    insertcolor=palette.fg, bordercolor=palette.border)
    style.configure("TCombobox", fieldbackground=palette.panel,
                    foreground=palette.fg, bordercolor=palette.border)
    style.configure("Treeview", background=palette.panel, fieldbackground=palette.panel,
                    foreground=palette.fg, rowheight=22, bordercolor=palette.border)
    style.configure("Treeview.Heading", background=palette.panel_alt,
                    foreground=palette.fg, font=("Segoe UI", font_size, "bold"))
    style.map("Treeview", background=[("selected", palette.accent)],
              foreground=[("selected", palette.accent_fg)])
    style.configure("TNotebook", background=palette.bg, bordercolor=palette.border)
    style.configure("TNotebook.Tab", background=palette.panel_alt,
                    foreground=palette.fg_dim, padding=(12, 6))
    style.map("TNotebook.Tab",
              background=[("selected", palette.panel)],
              foreground=[("selected", palette.fg)])
    style.configure("Horizontal.TProgressbar", background=palette.accent,
                    troughcolor=palette.panel_alt, bordercolor=palette.border)
    style.configure("TCheckbutton", background=palette.bg, foreground=palette.fg)
    style.configure("TRadiobutton", background=palette.bg, foreground=palette.fg)
    style.configure("TLabelframe", background=palette.bg, foreground=palette.fg,
                    bordercolor=palette.border)
    style.configure("TLabelframe.Label", background=palette.bg, foreground=palette.fg)
    root.option_add("*Font", base_font)
    root.option_add("*Text.Font", mono_font)
    root.option_add("*Menu.background", palette.panel)
    root.option_add("*Menu.foreground", palette.fg)
    root.option_add("*Menu.activeBackground", palette.accent)
    root.option_add("*Menu.activeForeground", palette.accent_fg)


def text_widget_kwargs(palette: Palette, *, code: bool = False,
                       font_size: int = 10) -> dict:
    """Keyword args for plain Tk Text widgets to match the theme."""
    return {
        "bg": palette.code_bg if code else palette.panel,
        "fg": palette.fg,
        "insertbackground": palette.fg,
        "selectbackground": palette.accent,
        "selectforeground": palette.accent_fg,
        "relief": "flat",
        "highlightthickness": 1,
        "highlightbackground": palette.border,
        "highlightcolor": palette.accent,
        "font": ("Consolas", font_size) if code else ("Segoe UI", font_size),
        "wrap": "word",
    }


def listbox_kwargs(palette: Palette, font_size: int = 10) -> dict:
    return {
        "bg": palette.panel,
        "fg": palette.fg,
        "selectbackground": palette.accent,
        "selectforeground": palette.accent_fg,
        "highlightthickness": 1,
        "highlightbackground": palette.border,
        "highlightcolor": palette.accent,
        "font": ("Segoe UI", font_size),
        "activestyle": "none",
    }
