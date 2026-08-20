from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from . import theme
from .audio import AudioEngine, AudioError, SAMPLE_RATE
from .cleanup import word_count
from .config import Settings, HISTORY_PATH
from .engine import TranscriptionEngine, READY, LOADING, FAILED, BUSY
from .hotkeys import HotkeyManager, label_for
from .inject import Injector, capture_focus, set_clipboard
from .overlay import Overlay

TYPING_WPM = 40.0


class App:
    def __init__(self, root):
        self.root = root
        self.settings = Settings()
        self.fonts = theme.resolve_fonts(root)
        theme.style_ttk(root)

        self.audio = AudioEngine(self.settings)
        self.engine = TranscriptionEngine(self.settings,
                                          on_status=self._engine_status,
                                          on_result=self._engine_result)
        self.injector = Injector(self.settings, own_hwnds=self._own_hwnds)
        self.overlay = Overlay(root, self.audio)
        self.hotkeys = HotkeyManager(
            self.settings,
            on_ptt_down=lambda: self._post(self._start_recording, "ptt"),
            on_ptt_up=lambda: self._post(self._stop_recording),
            on_toggle=lambda: self._post(self._toggle_recording),
            on_paste_last=lambda: self._post(self._paste_last),
            on_cancel=lambda: self._post(self._cancel_recording),
        )

        self._ui_queue = queue.Queue()
        self._history = []
        self._last_text = ""
        self._target = None
        self._recording = False
        self._toggle_mode = False
        self._session_words = 0
        self._session_count = 0
        self._session_audio = 0.0
        self._rtf_samples = []
        self._engine_state = ("idle", "")

        self._build()
        self._load_history()
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.root.after(60, self._pump)
        self.root.after(80, self._boot)

    # ------------------------------------------------------------------ ui
    def _build(self):
        r = self.root
        r.title("SraVaani Flow")
        r.configure(bg=theme.BG)
        r.minsize(1180, 760)
        self._fit_window(1360, 880)

        outer = ttk.Frame(r, style="TFrame")
        outer.pack(fill="both", expand=True)

        self._build_header(outer)
        ttk.Separator(outer, orient="horizontal").pack(fill="x")

        body = ttk.Frame(outer, style="TFrame")
        body.pack(fill="both", expand=True)

        self._build_sidebar(body)
        ttk.Separator(body, orient="vertical").pack(side="right", fill="y",
                                                    padx=(0, theme.PAD))

        main = ttk.Frame(body, style="TFrame")
        main.pack(side="left", fill="both", expand=True,
                  padx=(theme.PAD, 0), pady=(0, 0))
        self._build_tabbar(main)

        self._pages = {}
        self._page_host = ttk.Frame(main, style="TFrame")
        self._page_host.pack(fill="both", expand=True)
        for name, builder in (("Dictate", self._build_dictate_tab),
                              ("Notes", self._build_notes_tab),
                              ("Settings", self._build_settings_tab)):
            page = ttk.Frame(self._page_host, style="TFrame")
            self._pages[name] = page
            builder(page)

        ttk.Separator(outer, orient="horizontal").pack(fill="x")
        self._build_statusbar(outer)
        self.select_tab("Dictate")

    def _build_tabbar(self, parent):
        bar = ttk.Frame(parent, style="TFrame")
        bar.pack(fill="x", pady=(theme.GAP, 0))
        self._tab_buttons = {}
        self._tab_marks = {}
        self._active_tab = None
        for name in ("Dictate", "Notes", "Settings"):
            holder = ttk.Frame(bar, style="TFrame")
            holder.pack(side="left", padx=(0, 4))
            label = tk.Label(holder, text=name, bg=theme.BG, fg=theme.MUTED,
                             font=(theme.FONT_UI, 12), padx=16, pady=9,
                             cursor="hand2")
            label.pack(fill="x")
            mark = tk.Frame(holder, height=2, bg=theme.BG)
            mark.pack(fill="x")
            label.bind("<Button-1>", lambda e, n=name: self.select_tab(n))
            label.bind("<Enter>", lambda e, n=name: self._hover_tab(n, True))
            label.bind("<Leave>", lambda e, n=name: self._hover_tab(n, False))
            self._tab_buttons[name] = label
            self._tab_marks[name] = mark
        ttk.Separator(parent, orient="horizontal").pack(fill="x")

    def _hover_tab(self, name, entering):
        if name == self._active_tab:
            return
        self._tab_buttons[name].configure(
            fg=theme.FG_SOFT if entering else theme.MUTED)

    def select_tab(self, name):
        if name not in self._pages:
            return
        self._active_tab = name
        for other, page in self._pages.items():
            selected = (other == name)
            self._tab_buttons[other].configure(
                fg=theme.FG if selected else theme.MUTED,
                font=(theme.FONT_UI, 12, "bold") if selected else (theme.FONT_UI, 12))
            self._tab_marks[other].configure(bg=theme.INK if selected else theme.BG)
            if selected:
                page.pack(fill="both", expand=True)
            else:
                page.pack_forget()

    def _fit_window(self, want_w, want_h):
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            w = max(min(want_w, sw - 80), 900)
            h = max(min(want_h, sh - 120), 640)
            x = max((sw - w) // 2, 0)
            y = max((sh - h) // 3, 0)
            self.root.geometry("%dx%d+%d+%d" % (w, h, x, y))
        except Exception:
            self.root.geometry("%dx%d" % (want_w, want_h))

    def _build_header(self, parent):
        head = ttk.Frame(parent, style="TFrame")
        head.pack(fill="x", padx=theme.PAD, pady=(theme.PAD, theme.GAP))

        left = ttk.Frame(head, style="TFrame")
        left.pack(side="left")
        tk.Label(left, text="SraVaani Flow", bg=theme.BG, fg=theme.FG,
                 font=(theme.FONT_UI, 21, "bold")).pack(anchor="w")
        tk.Label(left, text="Offline dictation  ·  ARTPARK-IISc / SraVaani-1.0",
                 bg=theme.BG, fg=theme.MUTED,
                 font=(theme.FONT_UI, 10)).pack(anchor="w", pady=(2, 0))

        right = ttk.Frame(head, style="TFrame")
        right.pack(side="right")
        self.badge_device = tk.Label(right, text="—", bg=theme.BG,
                                     fg=theme.MUTED, font=(theme.FONT_MONO, 10),
                                     padx=11, pady=5, bd=1, relief="solid",
                                     highlightbackground=theme.BORDER)
        self.badge_device.pack(side="right", padx=(6, 0))
        self.badge_state = tk.Label(right, text="STARTING", bg=theme.SURFACE_ALT,
                                    fg=theme.FG_SOFT, font=(theme.FONT_MONO, 10, "bold"),
                                    padx=11, pady=5)
        self.badge_state.pack(side="right")

    def _build_dictate_tab(self, tab):

        bar = ttk.Frame(tab, style="TFrame")
        bar.pack(fill="x", pady=(theme.GAP, 6))
        tk.Label(bar, text="LATEST TRANSCRIPT", bg=theme.BG, fg=theme.MUTED,
                 font=(theme.FONT_UI, 10, "bold")).pack(side="left")
        self.hint_label = tk.Label(bar, text="", bg=theme.BG, fg=theme.MUTED,
                                   font=(theme.FONT_UI, 10))
        self.hint_label.pack(side="right")

        self.transcript = self._make_text(tab, height=7, size=17)
        self.transcript.pack(fill="both", expand=False)

        actions = ttk.Frame(tab, style="TFrame")
        actions.pack(fill="x", pady=(8, theme.GAP))
        ttk.Button(actions, text="Copy", command=self._copy_last).pack(side="left")
        ttk.Button(actions, text="Paste to last app",
                   command=self._paste_last).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Send to Note",
                   command=self._send_last_to_note).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Clear", command=self._clear_transcript).pack(side="left", padx=(6, 0))

        tk.Label(tab, text="HISTORY", bg=theme.BG, fg=theme.MUTED,
                 font=(theme.FONT_UI, 10, "bold")).pack(anchor="w", pady=(4, 6))

        wrap = ttk.Frame(tab, style="TFrame")
        wrap.pack(fill="both", expand=True, pady=(0, theme.PAD))
        self.history_list = tk.Listbox(
            wrap, bg=theme.SURFACE, fg=theme.FG_SOFT, bd=0, highlightthickness=1,
            highlightbackground=theme.BORDER, highlightcolor=theme.BORDER,
            selectbackground=theme.FG, selectforeground=theme.BG,
            font=(theme.FONT_INDIC, 12), activestyle="none")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.history_list.yview,
                           style="Vertical.TScrollbar")
        self.history_list.configure(yscrollcommand=sb.set)
        self.history_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.history_list.bind("<Double-Button-1>", self._history_activate)

    def _build_notes_tab(self, tab):

        bar = ttk.Frame(tab, style="TFrame")
        bar.pack(fill="x", pady=(theme.GAP, 6))
        self.note_title = tk.StringVar(value="Untitled note")
        entry = ttk.Entry(bar, textvariable=self.note_title, width=34)
        entry.pack(side="left")

        self.capture_to_note = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Dictate into this note",
                        variable=self.capture_to_note,
                        command=self._update_hints).pack(side="left", padx=(10, 0))

        self.note_stamp = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Timestamps",
                        variable=self.note_stamp).pack(side="left", padx=(10, 0))

        self.note_counter = tk.Label(bar, text="0 words", bg=theme.BG,
                                     fg=theme.MUTED, font=(theme.FONT_MONO, 10))
        self.note_counter.pack(side="right")

        self.note_hint = tk.Label(
            tab,
            text=("Open this tab and hold %s to dictate straight into the note. "
                  "Tick “Dictate into this note” to keep capturing here "
                  "even while another app is focused."
                  % label_for(self.settings.get("hotkey_ptt"))),
            bg=theme.BG, fg=theme.MUTED, font=(theme.FONT_UI, 10),
            anchor="w", justify="left", wraplength=760)
        self.note_hint.pack(fill="x", pady=(0, 8))

        self.note = self._make_text(tab, height=18, size=14, editable=True)
        self.note.pack(fill="both", expand=True)
        self.note.bind("<<Modified>>", self._note_modified)

        actions = ttk.Frame(tab, style="TFrame")
        actions.pack(fill="x", pady=(8, theme.PAD))
        ttk.Button(actions, text="Save as…", command=self._save_note).pack(side="left")
        ttk.Button(actions, text="Open…", command=self._open_note).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Copy all", command=self._copy_note).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="New", command=self._new_note).pack(side="left", padx=(6, 0))

    def _build_settings_tab(self, tab):

        grid = ttk.Frame(tab, style="TFrame")
        grid.pack(fill="both", expand=True, pady=theme.GAP)

        self._section(grid, "INPUT")
        row = ttk.Frame(grid, style="TFrame")
        row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text="Microphone", width=18).pack(side="left")
        self.device_names = ["System default"]
        self.device_ids = [None]
        for idx, name in AudioEngine.list_devices():
            self.device_names.append("%s" % name)
            self.device_ids.append(idx)
        self.device_var = tk.StringVar(value=self.device_names[0])
        current = self.settings.get("input_device")
        if current in self.device_ids:
            self.device_var.set(self.device_names[self.device_ids.index(current)])
        combo = ttk.Combobox(row, textvariable=self.device_var, values=self.device_names,
                             state="readonly", width=40)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self._change_device)

        self._toggle(grid, "Noise reduction (adaptive)", "denoise")
        self._toggle(grid, "High-pass filter (80 Hz)", "highpass")
        self._toggle(grid, "Trim silence with VAD", "vad_trim")
        self._toggle(grid, "Auto gain for quiet mics", "auto_gain")

        self._section(grid, "OUTPUT")
        self._toggle(grid, "Paste into the focused app", "auto_paste")
        self._toggle(grid, "Copy to clipboard", "auto_copy")
        self._toggle(grid, "Clean up fillers and casing", "cleanup")
        self._toggle(grid, "Spoken punctuation (\"comma\", \"full stop\")",
                     "spoken_punctuation")

        self._section(grid, "SHORTCUTS")
        self.hotkey_vars = {}
        for key, label, choices in (
                ("hotkey_ptt", "Hold to talk",
                 ["shift_r", "ctrl_r", "alt_r", "f8", "f7", "scroll_lock", "pause"]),
                ("hotkey_toggle", "Toggle dictation",
                 ["f9", "f10", "f7", "f8", "pause", "scroll_lock"]),
                ("hotkey_paste_last", "Paste last transcript",
                 ["f11", "f12", "f10", "insert"])):
            row = ttk.Frame(grid, style="TFrame")
            row.pack(fill="x", pady=(0, 6))
            ttk.Label(row, text=label, width=22).pack(side="left")
            var = tk.StringVar(value=str(self.settings.get(key)))
            box = ttk.Combobox(row, textvariable=var,
                               values=[label_for(c) for c in choices],
                               state="readonly", width=16)
            box.set(label_for(self.settings.get(key)))
            box.pack(side="left")
            box.bind("<<ComboboxSelected>>",
                     lambda e, k=key, c=choices, b=None, v=var: self._change_hotkey(k, c, v))
            self.hotkey_vars[key] = var

        self._section(grid, "COMPUTE")
        row = ttk.Frame(grid, style="TFrame")
        row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text="Device", width=18).pack(side="left")
        self.compute_var = tk.StringVar(value=str(self.settings.get("device", "auto")))
        cb = ttk.Combobox(row, textvariable=self.compute_var,
                          values=["auto", "cuda", "cpu"], state="readonly", width=12)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>",
                lambda e: self.settings.set("device", self.compute_var.get()))
        ttk.Label(row, text="restart required", style="Muted.TLabel").pack(side="left", padx=(8, 0))

        self._section(grid, "VOCABULARY")
        ttk.Label(grid, text="One term per line. Fixes domain spellings such as SraVaani, IISc.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 6))
        self.vocab = self._make_text(grid, height=6, size=12, editable=True, mono=True)
        self.vocab.pack(fill="x")
        self.vocab.insert("1.0", "\n".join(self.settings.get("vocabulary") or []))
        ttk.Button(grid, text="Save vocabulary",
                   command=self._save_vocab).pack(anchor="w", pady=(8, theme.PAD))

    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=theme.BG, fg=theme.MUTED,
                 font=(theme.FONT_UI, 10, "bold")).pack(anchor="w", pady=(theme.GAP, 6))

    def _toggle(self, parent, text, key):
        var = tk.BooleanVar(value=bool(self.settings.get(key)))

        def changed():
            self.settings.set(key, bool(var.get()))
            self._update_hints()

        ttk.Checkbutton(parent, text=text, variable=var,
                        command=changed).pack(anchor="w", pady=1)
        return var

    def _build_sidebar(self, parent):
        side = ttk.Frame(parent, style="TFrame", width=320)
        side.pack(side="right", fill="y", padx=(0, theme.PAD), pady=theme.GAP)
        side.pack_propagate(False)

        tk.Label(side, text="SHORTCUTS", bg=theme.BG, fg=theme.MUTED,
                 font=(theme.FONT_UI, 10, "bold")).pack(anchor="w")
        self.shortcut_rows = {}
        for key, desc in (("hotkey_ptt", "Hold to talk"),
                          ("hotkey_toggle", "Toggle dictation"),
                          ("hotkey_paste_last", "Paste last"),
                          ("esc", "Cancel")):
            row = ttk.Frame(side, style="TFrame")
            row.pack(fill="x", pady=3)
            name = "Esc" if key == "esc" else label_for(self.settings.get(key))
            chip = tk.Label(row, text=name, bg=theme.INK, fg=theme.ON_INK,
                            font=(theme.FONT_MONO, 10, "bold"), padx=10, pady=5)
            chip.pack(side="left")
            tk.Label(row, text=desc, bg=theme.BG, fg=theme.FG_SOFT,
                     font=(theme.FONT_UI, 10)).pack(side="left", padx=(8, 0))
            self.shortcut_rows[key] = chip

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=theme.GAP)

        tk.Label(side, text="INPUT LEVEL", bg=theme.BG, fg=theme.MUTED,
                 font=(theme.FONT_UI, 10, "bold")).pack(anchor="w")
        self.meter = tk.Canvas(side, height=44, bg=theme.SURFACE, bd=0,
                               highlightthickness=1, highlightbackground=theme.BORDER)
        self.meter.pack(fill="x", pady=(6, 0))
        self._meter_bars = []

        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=theme.GAP)

        tk.Label(side, text="SESSION", bg=theme.BG, fg=theme.MUTED,
                 font=(theme.FONT_UI, 10, "bold")).pack(anchor="w", pady=(0, 6))
        self.stat_vars = {}
        for key, label in (("words", "Words dictated"),
                           ("count", "Utterances"),
                           ("rtf", "Real-time factor"),
                           ("saved", "Typing time saved")):
            box = ttk.Frame(side, style="TFrame")
            box.pack(fill="x", pady=4)
            var = tk.StringVar(value="—")
            tk.Label(box, textvariable=var, bg=theme.BG, fg=theme.FG,
                     font=(theme.FONT_MONO, 24)).pack(anchor="w")
            tk.Label(box, text=label, bg=theme.BG, fg=theme.MUTED,
                     font=(theme.FONT_UI, 10)).pack(anchor="w")
            self.stat_vars[key] = var

    def _build_statusbar(self, parent):
        bar = ttk.Frame(parent, style="TFrame")
        bar.pack(fill="x", padx=theme.PAD, pady=8)
        self.status_var = tk.StringVar(value="Starting…")
        tk.Label(bar, textvariable=self.status_var, bg=theme.BG, fg=theme.FG_SOFT,
                 font=(theme.FONT_UI, 11)).pack(side="left")
        self.detail_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.detail_var, bg=theme.BG, fg=theme.MUTED,
                 font=(theme.FONT_MONO, 10)).pack(side="right")

    def _make_text(self, parent, height=8, size=14, editable=False, mono=False):
        family = theme.FONT_MONO if mono else theme.FONT_INDIC
        widget = tk.Text(parent, height=height, wrap="word", bd=0,
                         bg=theme.SURFACE, fg=theme.FG,
                         insertbackground=theme.FG,
                         selectbackground=theme.SELECT, selectforeground=theme.FG,
                         highlightthickness=1, highlightbackground=theme.BORDER,
                         highlightcolor=theme.BORDER_STRONG,
                         font=(family, size), padx=14, pady=12,
                         spacing1=2, spacing3=4)
        if not editable:
            widget.configure(state="disabled")
        widget.tag_configure("meta", foreground=theme.MUTED,
                             font=(theme.FONT_MONO, 10))
        return widget

    # -------------------------------------------------------------- startup
    def _boot(self):
        self.engine.start()
        try:
            self.audio.start()
        except AudioError as exc:
            self._set_status("Microphone unavailable", str(exc))
            messagebox.showerror("Microphone", str(exc))
        if not self.hotkeys.start():
            self._set_status("Hotkeys unavailable", self.hotkeys.last_error or "")
        self._update_hints()
        self._tick()

    def _own_hwnds(self):
        out = set()
        for win in (self.root, self.overlay.win):
            try:
                out.add(win.winfo_id())
                import win32gui
                parent = win32gui.GetParent(win.winfo_id())
                if parent:
                    out.add(parent)
            except Exception:
                pass
        return out

    # ------------------------------------------------------------ threading
    def _post(self, fn, *args):
        self._ui_queue.put((fn, args))

    def _pump(self):
        try:
            while True:
                fn, args = self._ui_queue.get_nowait()
                try:
                    fn(*args)
                except Exception:
                    import traceback
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.root.after(40, self._pump)

    def _tick(self):
        self._draw_meter()
        if self._recording:
            elapsed = self.audio.elapsed
            self.status_var.set("Listening   %.1fs" % elapsed)
            limit = float(self.settings.get("max_record_seconds", 120))
            if elapsed > limit:
                self._stop_recording()
        self.root.after(50, self._tick)

    def _draw_meter(self):
        try:
            c = self.meter
            width = c.winfo_width()
            if width < 10:
                return
            values = list(self.audio.waveform)
            n = 40
            if not self._meter_bars or len(self._meter_bars) != n:
                c.delete("all")
                self._meter_bars = []
                step = width / float(n)
                for i in range(n):
                    x = i * step
                    self._meter_bars.append(
                        c.create_rectangle(x + 1, 16, x + step - 1, 18,
                                           fill=theme.ACCENT_DIM, outline=""))
            stride = max(len(values) / float(n), 1e-6)
            for i, bar in enumerate(self._meter_bars):
                v = values[min(int(i * stride), len(values) - 1)] if values else 0.0
                h = max(1.0, min(v, 1.0) * 14.0)
                coords = c.coords(bar)
                if coords:
                    c.coords(bar, coords[0], 17 - h, coords[2], 17 + h)
                    c.itemconfigure(bar, fill=theme.FG if v > 0.18 else theme.ACCENT_DIM)
        except Exception:
            pass

    # ------------------------------------------------------------ recording
    def _ready(self):
        return self.engine.status in (READY, BUSY) and self.engine.model is not None

    def _start_recording(self, source="ptt"):
        if self._recording:
            return
        if not self._ready():
            self.overlay.flash("empty", "Model still loading", 1100)
            return
        if not self.audio.running:
            try:
                self.audio.start()
            except AudioError as exc:
                self._set_status("Microphone unavailable", str(exc))
                return
        self._target = capture_focus()
        self._recording = True
        self._toggle_mode = (source == "toggle")
        self.audio.begin()
        hint = self._target.title[:44] if self._target and self._target.title else ""
        self.overlay.show("listening", hint)

    def _stop_recording(self):
        if not self._recording:
            return
        self._recording = False
        self._toggle_mode = False
        audio = self.audio.end()
        seconds = audio.size / float(SAMPLE_RATE)
        minimum = float(self.settings.get("min_record_seconds", 0.35))
        if seconds < minimum:
            self.overlay.flash("empty", "Too short", 800)
            self._set_status("Ready", "clip too short (%.2fs)" % seconds)
            return
        self.overlay.set_state("transcribing", "")
        self.status_var.set("Transcribing %.1fs…" % seconds)
        target = self._target
        threading.Thread(target=self._process_and_submit,
                         args=(audio, target), daemon=True).start()

    def _process_and_submit(self, audio, target):
        try:
            processed, info = self.audio.process(audio)
            self.engine.submit(processed, info, target=target)
        except Exception as exc:
            self._post(self._set_status, "Audio processing failed", str(exc))
            self._post(self.overlay.hide)

    def _toggle_recording(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording("toggle")

    def _cancel_recording(self):
        if not self._recording:
            return
        self._recording = False
        self.audio.cancel()
        self.overlay.flash("cancelled", "", 700)
        self._set_status("Ready", "cancelled")

    # -------------------------------------------------------------- results
    def _engine_status(self, status, detail):
        self._post(self._apply_engine_status, status, detail)

    def _apply_engine_status(self, status, detail):
        self._engine_state = (status, detail)
        labels = {LOADING: "LOADING", READY: "READY", BUSY: "BUSY", FAILED: "ERROR"}
        self.badge_state.configure(text=labels.get(status, status.upper()),
                                   fg=theme.ON_INK if status == READY else theme.FG_SOFT,
                                   bg=theme.INK if status == READY else theme.SURFACE_ALT)
        if status == READY:
            self.badge_device.configure(text=detail)
            self._set_status("Ready", "hold %s to dictate"
                             % label_for(self.settings.get("hotkey_ptt")))
        elif status == LOADING:
            self._set_status(detail or "Loading model…", "")
        elif status == FAILED:
            self._set_status("Model failed to load", detail)
            messagebox.showerror("SraVaani Flow", detail)
        elif status == BUSY:
            self._set_status(detail or "Transcribing…", "")

    def _engine_result(self, result):
        self._post(self._apply_result, result)

    def _apply_result(self, result):
        if result.error == "no_speech":
            self.overlay.flash("empty", "No speech detected", 900)
            self._set_status("Ready", "no speech detected")
            return
        if result.error == "too_short":
            self.overlay.flash("empty", "Too short", 800)
            return
        if result.error:
            self.overlay.hide()
            self._set_status("Transcription failed", str(result.error)[:120])
            return
        if not result.text:
            self.overlay.flash("empty", "Nothing recognised", 900)
            self._set_status("Ready", "empty result")
            return

        self.overlay.hide()
        self._last_text = result.text
        self._show_transcript(result)
        self._record_stats(result)

        entry = {"time": time.time(), "text": result.text, "raw": result.raw,
                 "seconds": result.job.info.get("seconds", 0.0),
                 "elapsed": result.elapsed}
        self._history.insert(0, entry)
        self.history_list.insert(0, self._history_line(entry))
        self._append_history_file(entry)

        dictating_into_note = (
            self.capture_to_note.get()
            or (self._active_tab == "Notes"
                and self.injector.is_own_window(result.job.target)))
        if dictating_into_note:
            self._append_to_note(result.text)
            self._set_status("Ready", "added to note  ·  %.2fs  ·  %d words"
                             % (result.elapsed, result.words))
            set_clipboard(result.text)
            return

        threading.Thread(target=self._deliver, args=(result,), daemon=True).start()

    def _deliver(self, result):
        outcome = self.injector.deliver(result.text, result.job.target)
        self._post(self._report_delivery, result, outcome)

    def _report_delivery(self, result, outcome):
        bits = ["%.2fs" % result.elapsed, "%d words" % result.words]
        if outcome.get("pasted"):
            name = outcome.get("target") or "focused app"
            bits.append("pasted into %s" % name[:32])
        elif outcome.get("copied"):
            reason = {"own_window": "app focused, copied instead",
                      "no_target": "no target window, copied",
                      "focus_failed": "could not refocus, copied",
                      "paste_disabled": "copied"}.get(outcome.get("reason"), "copied")
            bits.append(reason)
        self._set_status("Ready", "  ·  ".join(bits))

    def _show_transcript(self, result):
        w = self.transcript
        w.configure(state="normal")
        w.delete("1.0", "end")
        w.insert("1.0", result.text)
        info = result.job.info or {}
        meta = "%.2fs  ·  %.1fx real time  ·  %d words" % (
            result.elapsed,
            (info.get("seconds", 0.0) / result.elapsed) if result.elapsed else 0.0,
            result.words)
        if info.get("denoised"):
            meta += "  ·  denoised"
        if info.get("snr_db") is not None:
            meta += "  ·  SNR %.0f dB" % info["snr_db"]
        w.insert("end", "\n\n" + meta, "meta")
        w.configure(state="disabled")

    def _record_stats(self, result):
        self._session_words += result.words
        self._session_count += 1
        self._session_audio += (result.job.info or {}).get("seconds", 0.0)
        if result.elapsed > 0:
            seconds = (result.job.info or {}).get("seconds", 0.0)
            if seconds > 0:
                self._rtf_samples.append(result.elapsed / seconds)
        self.stat_vars["words"].set(str(self._session_words))
        self.stat_vars["count"].set(str(self._session_count))
        if self._rtf_samples:
            avg = sum(self._rtf_samples) / len(self._rtf_samples)
            self.stat_vars["rtf"].set("%.2f" % avg)
        typing_seconds = self._session_words / TYPING_WPM * 60.0
        saved = max(typing_seconds - self._session_audio, 0.0)
        self.stat_vars["saved"].set("%dm %02ds" % (int(saved // 60), int(saved % 60)))

    # -------------------------------------------------------------- history
    def _history_line(self, entry):
        stamp = time.strftime("%H:%M:%S", time.localtime(entry["time"]))
        text = " ".join(entry["text"].split())
        return "%s   %s" % (stamp, text[:110])

    def _load_history(self):
        try:
            if not HISTORY_PATH.exists():
                return
            lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()[-200:]
            for line in lines:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("text"):
                    self._history.insert(0, entry)
            for entry in self._history:
                self.history_list.insert("end", self._history_line(entry))
        except Exception:
            pass

    def _append_history_file(self, entry):
        def write():
            try:
                with open(HISTORY_PATH, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass
        threading.Thread(target=write, daemon=True).start()

    def _history_activate(self, _event=None):
        sel = self.history_list.curselection()
        if not sel:
            return
        entry = self._history[sel[0]]
        self._last_text = entry["text"]
        set_clipboard(entry["text"])
        self._set_status("Ready", "copied from history")

    # ---------------------------------------------------------------- notes
    def _append_to_note(self, text):
        widget = self.note
        current = widget.get("1.0", "end-1c")
        prefix = ""
        if current.strip():
            prefix = "\n\n"
        if self.note_stamp.get():
            prefix += "[%s]  " % time.strftime("%H:%M")
        widget.insert("end", prefix + text)
        widget.see("end")
        self._update_note_counter()
        self.select_tab("Notes")

    def _note_modified(self, _event=None):
        try:
            self.note.edit_modified(False)
        except Exception:
            pass
        self._update_note_counter()

    def _update_note_counter(self):
        try:
            text = self.note.get("1.0", "end-1c")
            self.note_counter.configure(text="%d words" % word_count(text))
        except Exception:
            pass

    def _save_note(self):
        text = self.note.get("1.0", "end-1c")
        if not text.strip():
            messagebox.showinfo("Notes", "This note is empty.")
            return
        name = (self.note_title.get() or "note").strip().replace(" ", "-")
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", initialfile="%s.txt" % name,
            filetypes=[("Text", "*.txt"), ("Markdown", "*.md"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            self._set_status("Ready", "note saved")
        except Exception as exc:
            messagebox.showerror("Notes", str(exc))

    def _open_note(self):
        path = filedialog.askopenfilename(
            filetypes=[("Text", "*.txt *.md"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            self.note.delete("1.0", "end")
            self.note.insert("1.0", content)
            self.note_title.set(path.rsplit("\\", 1)[-1].rsplit(".", 1)[0])
            self._update_note_counter()
        except Exception as exc:
            messagebox.showerror("Notes", str(exc))

    def _copy_note(self):
        text = self.note.get("1.0", "end-1c")
        if text.strip():
            set_clipboard(text)
            self._set_status("Ready", "note copied")

    def _new_note(self):
        if self.note.get("1.0", "end-1c").strip():
            if not messagebox.askyesno("Notes", "Discard the current note?"):
                return
        self.note.delete("1.0", "end")
        self.note_title.set("Untitled note")
        self._update_note_counter()

    def _send_last_to_note(self):
        if not self._last_text:
            return
        self._append_to_note(self._last_text)

    # -------------------------------------------------------------- actions
    def _copy_last(self):
        if self._last_text:
            set_clipboard(self._last_text)
            self._set_status("Ready", "copied")

    def _paste_last(self):
        if not self._last_text:
            return
        target = self._target
        threading.Thread(
            target=lambda: self.injector.deliver(self._last_text, target),
            daemon=True).start()

    def _clear_transcript(self):
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self._last_text = ""

    def _change_device(self, _event=None):
        try:
            idx = self.device_names.index(self.device_var.get())
        except ValueError:
            return
        self.settings.set("input_device", self.device_ids[idx])
        try:
            self.audio.start()
            self._set_status("Ready", "microphone switched")
        except AudioError as exc:
            messagebox.showerror("Microphone", str(exc))

    def _change_hotkey(self, key, choices, var):
        labels = [label_for(c) for c in choices]
        try:
            chosen = choices[labels.index(var.get())]
        except ValueError:
            return
        for other in ("hotkey_ptt", "hotkey_toggle", "hotkey_paste_last"):
            if other != key and self.settings.get(other) == chosen:
                messagebox.showwarning(
                    "Shortcuts",
                    "%s is already used for another action." % label_for(chosen))
                var.set(label_for(self.settings.get(key)))
                return
        self.settings.set(key, chosen)
        self.hotkeys.refresh()
        chip = self.shortcut_rows.get(key)
        if chip is not None:
            chip.configure(text=label_for(chosen))
        self._update_hints()
        self._set_status("Ready", "shortcut updated")

    def _save_vocab(self):
        text = self.vocab.get("1.0", "end-1c")
        terms = [t.strip() for t in text.splitlines() if t.strip()]
        self.settings.set("vocabulary", terms)
        self._set_status("Ready", "%d vocabulary terms saved" % len(terms))

    def _update_hints(self):
        key = label_for(self.settings.get("hotkey_ptt"))
        if self.capture_to_note.get():
            self.hint_label.configure(text="Hold %s — text goes into the note" % key)
        elif self.settings.get("auto_paste", True):
            self.hint_label.configure(text="Hold %s — text lands at your cursor" % key)
        else:
            self.hint_label.configure(text="Hold %s — text is copied" % key)

    def _set_status(self, text, detail=""):
        self.status_var.set(text)
        self.detail_var.set(detail)

    # ------------------------------------------------------------- shutdown
    def shutdown(self):
        try:
            self.hotkeys.stop()
        except Exception:
            pass
        try:
            self.audio.stop()
        except Exception:
            pass
        try:
            self.engine.shutdown()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def enable_dpi_awareness():
    try:
        import ctypes
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            return
        except Exception:
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def main():
    enable_dpi_awareness()
    root = tk.Tk()
    try:
        dpi = float(root.winfo_fpixels("1i"))
        if dpi > 0:
            root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass
    App(root)
    root.mainloop()
