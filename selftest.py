from __future__ import annotations

import os
import sys
import time
import tkinter as tk

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sravaani_flow import theme
from sravaani_flow.app import App
from sravaani_flow.audio import AudioEngine, trim_to_speech, estimate_snr_db
from sravaani_flow.cleanup import clean, clean_hypothesis, segment_by_pauses
from sravaani_flow.config import Settings, SAMPLE_RATE
from sravaani_flow.engine import READY, FAILED

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("[%s] %-42s %s" % (PASS if ok else FAIL, name, detail))
    return ok


def tone_speech(seconds=2.0):
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), dtype=np.float32)
    env = (np.abs(np.sin(2 * np.pi * 2.5 * t)) > 0.35).astype(np.float32)
    sig = 0.0
    for f in (140, 280, 560, 1120):
        sig = sig + np.sin(2 * np.pi * f * t) / (f / 140.0)
    return (0.25 * sig * env).astype(np.float32)


def main():
    print("=" * 72)
    print("SraVaani Flow self-test")
    print("=" * 72)

    check("settings load", isinstance(Settings().as_dict(), dict))

    devices = AudioEngine.list_devices()
    check("input devices found", len(devices) > 0, "%d devices" % len(devices))

    silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    _, ratio = trim_to_speech(silence)
    check("VAD rejects silence", ratio == 0.0, "speech_ratio=%.2f" % ratio)

    speech = tone_speech()
    _, ratio2 = trim_to_speech(speech)
    check("VAD accepts signal", ratio2 > 0.0, "speech_ratio=%.2f" % ratio2)

    quiet = estimate_snr_db(speech, np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.0005)
    check("SNR estimate sane", quiet > 25, "%.1f dB" % quiet)

    txt = clean("um so basically the report is uh ready comma and i think we should send it full stop",
                vocabulary=["SraVaani"])
    check("cleanup english", txt == "So the report is ready, and I think we should send it.", txt)

    kn = "ನಮಸ್ಕಾರ"
    check("cleanup preserves Kannada", clean(kn) == kn)

    seg = segment_by_pauses([
        {"word": "hello", "start": 0.0, "end": 1.6},
        {"word": "world", "start": 1.6, "end": 2.1},
    ])
    check("pause segmentation", "." in seg, seg)

    root = tk.Tk()
    root.withdraw()
    fonts = theme.resolve_fonts(root)
    check("indic font available", fonts["indic"] in ("Nirmala UI", "Noto Sans"),
          fonts["indic"])

    app = App(root)
    root.deiconify()
    for _ in range(20):
        root.update()
        time.sleep(0.02)
    check("UI builds", len(app._pages) == 3, "%d tabs" % len(app._pages))
    check("overlay created", app.overlay.win is not None)
    check("hotkeys listening", app.hotkeys.running,
          app.hotkeys.last_error or "ok")
    check("audio stream running", app.audio.running,
          app.audio.last_error or "ok")

    print("\nwaiting for model to load…")
    deadline = time.time() + 180
    while time.time() < deadline:
        root.update()
        if app.engine.status in (READY, FAILED):
            break
        time.sleep(0.05)
    check("model ready", app.engine.status == READY,
          "%s %s" % (app.engine.status, app.engine.detail))

    if app.engine.status == READY:
        check("running on GPU", app.engine.device == "cuda",
              "%s / %s" % (app.engine.device, app.engine.precision))

        got = {}
        original = app._apply_result

        def capture(result):
            got["result"] = result
            original(result)

        app._apply_result = capture

        app.settings.set("auto_paste", False)
        processed, info = app.audio.process(tone_speech(1.5))
        app.engine.submit(processed, info, target=None)
        deadline = time.time() + 60
        while time.time() < deadline and "result" not in got:
            root.update()
            time.sleep(0.05)
        check("pipeline returns a result", "result" in got,
              repr(getattr(got.get("result"), "text", ""))[:60])

        got.clear()
        sil_proc, sil_info = app.audio.process(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
        app.engine.submit(sil_proc, sil_info, target=None)
        deadline = time.time() + 60
        while time.time() < deadline and "result" not in got:
            root.update()
            time.sleep(0.05)
        r = got.get("result")
        check("silence produces no text", r is not None and not r.text,
              "error=%s text=%r" % (getattr(r, "error", None), getattr(r, "text", None)))

        wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.wav")
        if os.path.exists(wav):
            import soundfile as sf
            data, sr = sf.read(wav, dtype="float32")
            if data.ndim > 1:
                data = data.mean(1)
            hyp = app.engine.model.transcribe([data], return_hypotheses=True,
                                              timestamps=True)[0]
            out = clean_hypothesis(hyp, vocabulary=app.settings.get("vocabulary"))
            check("real speech transcribed", bool(out.strip()), out[:80])

        app.settings.set("auto_paste", True)

    app.shutdown()

    print("\n" + "=" * 72)
    failed = [n for n, ok, _ in results if not ok]
    print("%d/%d checks passed" % (len(results) - len(failed), len(results)))
    if failed:
        print("FAILED: " + ", ".join(failed))
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
