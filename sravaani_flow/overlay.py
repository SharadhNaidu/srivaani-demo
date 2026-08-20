from __future__ import annotations

import tkinter as tk

from .theme import (BG, BORDER, FG, MUTED, FONT_UI, FONT_MONO, ACCENT_DIM,
                    INK, ON_INK)

WIDTH = 460
HEIGHT = 92
BOTTOM_MARGIN = 96
BAR_COUNT = 56


def _make_click_through(window):
    try:
        import win32con
        import win32gui
        hwnd = int(window.frame(), 16) if isinstance(window.frame(), str) else window.winfo_id()
        try:
            hwnd = win32gui.GetParent(window.winfo_id()) or window.winfo_id()
        except Exception:
            hwnd = window.winfo_id()
        styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        styles |= win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOOLWINDOW
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
        return True
    except Exception:
        return False


class Overlay:
    def __init__(self, root, audio):
        self.root = root
        self.audio = audio
        self.visible = False
        self._state = "listening"
        self._job = None

        self.win = tk.Toplevel(root)
        self.win.withdraw()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        try:
            self.win.attributes("-alpha", 0.97)
        except Exception:
            pass

        self.canvas = tk.Canvas(self.win, width=WIDTH, height=HEIGHT, bg=BG,
                                highlightthickness=2, highlightbackground=INK,
                                bd=0)
        self.canvas.pack(fill="both", expand=True)

        self._label = self.canvas.create_text(
            24, 28, text="LISTENING", anchor="w", fill=FG,
            font=(FONT_UI, 11, "bold"))
        self._timer = self.canvas.create_text(
            WIDTH - 24, 28, text="0.0s", anchor="e", fill=MUTED,
            font=(FONT_MONO, 11))
        self._hint = self.canvas.create_text(
            24, HEIGHT - 18, text="", anchor="w", fill=MUTED,
            font=(FONT_UI, 10))

        self._bars = []
        span = WIDTH - 48
        step = span / float(BAR_COUNT)
        for i in range(BAR_COUNT):
            x = 24 + i * step
            bar = self.canvas.create_rectangle(
                x, 54, x + max(step - 2.0, 1.0), 56,
                fill=ACCENT_DIM, outline="")
            self._bars.append(bar)

    def _position(self):
        try:
            sw = self.win.winfo_screenwidth()
            sh = self.win.winfo_screenheight()
            x = int((sw - WIDTH) / 2)
            y = int(sh - HEIGHT - BOTTOM_MARGIN)
            self.win.geometry("%dx%d+%d+%d" % (WIDTH, HEIGHT, x, y))
        except Exception:
            pass

    def show(self, state="listening", hint=""):
        self._state = state
        self._position()
        self.canvas.itemconfigure(self._hint, text=hint)
        self._apply_state()
        if not self.visible:
            self.win.deiconify()
            self.win.attributes("-topmost", True)
            _make_click_through(self.win)
            self.visible = True
        self._tick()

    def set_state(self, state, hint=""):
        self._state = state
        self.canvas.itemconfigure(self._hint, text=hint)
        self._apply_state()

    def _apply_state(self):
        text = {"listening": "LISTENING",
                "transcribing": "TRANSCRIBING",
                "cancelled": "CANCELLED",
                "empty": "NO SPEECH DETECTED"}.get(self._state, self._state.upper())
        self.canvas.itemconfigure(self._label, text=text)

    def hide(self):
        if self._job is not None:
            try:
                self.root.after_cancel(self._job)
            except Exception:
                pass
            self._job = None
        if self.visible:
            try:
                self.win.withdraw()
            except Exception:
                pass
            self.visible = False

    def flash(self, state, hint="", ms=900):
        self.set_state(state, hint)
        self.root.after(ms, self.hide)

    def _tick(self):
        if not self.visible:
            return
        try:
            if self._state == "listening":
                values = list(self.audio.waveform)
                self.canvas.itemconfigure(self._timer,
                                          text="%.1fs" % self.audio.elapsed)
            else:
                values = [0.0] * BAR_COUNT

            n = len(self._bars)
            if values:
                stride = max(len(values) / float(n), 1e-6)
            mid = 55.0
            for i, bar in enumerate(self._bars):
                v = values[min(int(i * stride), len(values) - 1)] if values else 0.0
                h = max(1.5, min(v, 1.0) * 24.0)
                coords = self.canvas.coords(bar)
                if coords:
                    self.canvas.coords(bar, coords[0], mid - h, coords[2], mid + h)
                    shade = FG if v > 0.18 else ACCENT_DIM
                    self.canvas.itemconfigure(bar, fill=shade)
        except Exception:
            pass
        self._job = self.root.after(40, self._tick)
