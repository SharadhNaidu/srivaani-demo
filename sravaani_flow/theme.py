from __future__ import annotations

import tkinter.font as tkfont

BG = "#FFFFFF"
SURFACE = "#FCFCFC"
SURFACE_ALT = "#F4F4F4"
BORDER = "#E6E6E6"
BORDER_STRONG = "#C4C4C4"
FG = "#0A0A0A"
FG_SOFT = "#3A3A3A"
MUTED = "#7A7A7A"
FAINT = "#BFBFBF"
ACCENT_DIM = "#D9D9D9"
SELECT = "#E8E8E8"
INK = "#0A0A0A"
ON_INK = "#FFFFFF"

FONT_UI = "Segoe UI"
FONT_MONO = "Consolas"
FONT_INDIC = "Nirmala UI"

PAD = 20
GAP = 12


def pick_font(root, candidates, fallback):
    try:
        available = set(tkfont.families(root))
    except Exception:
        return fallback
    for name in candidates:
        if name in available:
            return name
    return fallback


def resolve_fonts(root):
    global FONT_UI, FONT_MONO, FONT_INDIC
    FONT_UI = pick_font(root, ["Segoe UI Variable Text", "Segoe UI", "Inter", "Arial"], "Arial")
    FONT_MONO = pick_font(root, ["Cascadia Mono", "Consolas", "JetBrains Mono", "Courier New"],
                          "Courier New")
    FONT_INDIC = pick_font(root, ["Nirmala UI", "Noto Sans", "Segoe UI"], FONT_UI)
    return {"ui": FONT_UI, "mono": FONT_MONO, "indic": FONT_INDIC}


def style_ttk(root):
    from tkinter import ttk
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(".", background=BG, foreground=FG, borderwidth=0,
                    focuscolor=BG, font=(FONT_UI, 11))
    style.configure("TFrame", background=BG)
    style.configure("Surface.TFrame", background=SURFACE)
    style.configure("TLabel", background=BG, foreground=FG_SOFT, font=(FONT_UI, 11))
    style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=(FONT_UI, 10))
    style.configure("Head.TLabel", background=BG, foreground=FG, font=(FONT_UI, 12, "bold"))
    style.configure("Mono.TLabel", background=BG, foreground=FG_SOFT, font=(FONT_MONO, 11))

    style.configure("TButton", background=BG, foreground=FG,
                    bordercolor=BORDER_STRONG, lightcolor=BG, darkcolor=BG,
                    relief="flat", padding=(14, 8), font=(FONT_UI, 11))
    style.map("TButton",
              background=[("active", SURFACE_ALT), ("pressed", SELECT),
                          ("disabled", BG)],
              bordercolor=[("active", FG), ("disabled", BORDER)],
              foreground=[("disabled", FAINT)])

    style.configure("Primary.TButton", background=INK, foreground=ON_INK,
                    bordercolor=INK, lightcolor=INK, darkcolor=INK,
                    font=(FONT_UI, 11, "bold"), padding=(16, 9))
    style.map("Primary.TButton",
              background=[("active", "#262626"), ("pressed", "#3A3A3A"),
                          ("disabled", SURFACE_ALT)],
              foreground=[("disabled", FAINT)])

    style.configure("TCheckbutton", background=BG, foreground=FG_SOFT,
                    font=(FONT_UI, 11), indicatorcolor=BG,
                    bordercolor=BORDER_STRONG, focuscolor=BG,
                    indicatormargin=6, padding=(0, 3))
    style.map("TCheckbutton",
              background=[("active", BG)],
              indicatorcolor=[("selected", INK), ("!selected", BG)],
              bordercolor=[("active", FG)],
              foreground=[("active", FG)])

    style.configure("TCombobox", fieldbackground=BG, background=BG,
                    foreground=FG, arrowcolor=FG_SOFT, bordercolor=BORDER_STRONG,
                    lightcolor=BG, darkcolor=BG,
                    selectbackground=BG, selectforeground=FG,
                    padding=(10, 7))
    style.map("TCombobox",
              fieldbackground=[("readonly", BG)],
              bordercolor=[("focus", FG), ("active", FG)],
              foreground=[("readonly", FG)])

    style.configure("TNotebook", background=BG, borderwidth=0,
                    tabmargins=(0, 0, 0, 0), lightcolor=BG, darkcolor=BG,
                    bordercolor=BG)
    style.configure("TNotebook.Tab", background=BG, foreground=MUTED,
                    padding=(2, 10), borderwidth=0, font=(FONT_UI, 11),
                    lightcolor=BG, darkcolor=BG, bordercolor=BG,
                    focuscolor=BG)
    style.map("TNotebook.Tab",
              background=[("selected", BG), ("active", BG)],
              lightcolor=[("selected", BG)], darkcolor=[("selected", BG)],
              bordercolor=[("selected", BG)],
              expand=[("selected", (0, 0, 0, 0))],
              foreground=[("selected", FG), ("active", FG_SOFT)],
              font=[("selected", (FONT_UI, 11, "bold"))])

    style.configure("Vertical.TScrollbar", background=ACCENT_DIM, troughcolor=BG,
                    bordercolor=BG, arrowcolor=MUTED, width=11, relief="flat")
    style.map("Vertical.TScrollbar", background=[("active", MUTED)])
    style.configure("TSeparator", background=BORDER)
    style.configure("TEntry", fieldbackground=BG, foreground=FG,
                    bordercolor=BORDER_STRONG, lightcolor=BG, darkcolor=BG,
                    insertcolor=FG, padding=(10, 7))
    style.map("TEntry", bordercolor=[("focus", FG)])
    return style
