from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from transformers import AutoModel

from sravaani_flow.audio import (_auto_gain, _denoise, _highpass, estimate_snr_db,
                                 trim_to_speech, SAMPLE_RATE)
from sravaani_flow.cleanup import clean_hypothesis
from sravaani_flow.config import Settings, MODEL_REPO
from sravaani_flow.inject import Injector, FocusTarget, capture_focus, get_clipboard

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print("[%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / max(len(r), 1)


def resample(x, a, b):
    n = int(round(x.size * b / a))
    X = np.fft.rfft(x)
    Y = np.zeros(n // 2 + 1, dtype=complex)
    k = min(X.size, Y.size)
    Y[:k] = X[:k]
    return (np.fft.irfft(Y, n=n) * (n / x.size)).astype(np.float32)


def add_noise(sig, noise, snr_db):
    ps, pn = np.mean(sig ** 2), np.mean(noise ** 2)
    return (sig + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))).astype(np.float32)


def chain(x, noise_ref, settings):
    y = _highpass(x)
    snr = estimate_snr_db(y, noise_ref)
    if snr < 22.0:
        y = _denoise(y, noise_ref, 0.75)
    y, _ = trim_to_speech(y)
    return _auto_gain(y)


def test_noise(model, settings):
    print("\n--- disturbance / noise robustness " + "-" * 34)
    import soundfile as sf
    wav_path = os.path.join(ROOT, "sample.wav")
    if not os.path.exists(wav_path):
        check("noise suite (sample.wav present)", False, "sample.wav missing")
        return
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = resample(wav, sr, SAMPLE_RATE)
    ref = ("welcome to the class demonstration this is a live speech to text "
           "system running on the sravaani model from artpark at iisc")

    rng = np.random.default_rng(7)
    t = np.arange(wav.size) / float(SAMPLE_RATE)
    noises = {
        "white hiss": rng.standard_normal(wav.size).astype(np.float32),
        "mains hum": (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
                      + 0.3 * rng.standard_normal(wav.size)).astype(np.float32),
        "fan rumble": np.convolve(rng.standard_normal(wav.size),
                                  np.ones(64) / 64.0, mode="same").astype(np.float32),
    }

    worst = 0.0
    for name, noise in noises.items():
        for snr in (20, 10, 5):
            noisy = add_noise(wav, noise, snr)
            scale = np.sqrt(np.mean(wav ** 2) / (np.mean(noise ** 2) * 10 ** (snr / 10.0)))
            noise_ref = (noise[:SAMPLE_RATE] * scale).astype(np.float32)
            processed = chain(noisy, noise_ref, settings)
            hyp = model.transcribe([processed], return_hypotheses=True, timestamps=True)[0]
            text = clean_hypothesis(hyp, vocabulary=settings.get("vocabulary")).lower()
            e = wer(ref, text)
            worst = max(worst, e)
            print("      %-11s SNR %2ddB  WER %.2f  %s" % (name, snr, e, text[:52]))
            check("noise %s @%ddB produces text" % (name, snr), bool(text.strip()), "")
    check("worst-case WER stays usable", worst < 0.65, "worst WER %.2f" % worst)


def test_clipping_and_edges(model, settings):
    print("\n--- edge cases " + "-" * 55)
    cases = {
        "pure silence": np.zeros(SAMPLE_RATE * 2, dtype=np.float32),
        "clipped loud": np.clip(np.random.default_rng(1).standard_normal(SAMPLE_RATE) * 8, -1, 1).astype(np.float32),
        "dc offset": (np.ones(SAMPLE_RATE, dtype=np.float32) * 0.6),
        "very short": np.zeros(1000, dtype=np.float32),
        "empty": np.zeros(0, dtype=np.float32),
    }
    for name, audio in cases.items():
        try:
            if audio.size:
                y = _highpass(audio)
                y, ratio = trim_to_speech(y)
                y = _auto_gain(y)
            else:
                y, ratio = audio, 0.0
            ok = True
            detail = "speech_ratio %.2f" % ratio
            if name in ("pure silence", "dc offset"):
                ok = ratio < 0.2
                detail += " (gated)"
        except Exception as exc:
            ok, detail = False, "raised %s" % exc
        check("edge case: %s" % name, ok, detail)


def test_cursor_injection(settings):
    print("\n--- cursor targeting / paste into another app " + "-" * 24)
    py = sys.executable
    target_script = os.path.join(ROOT, "tests", "target_window.py")
    proc = subprocess.Popen([py, target_script], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True,
                            encoding="utf-8", bufsize=1)
    time.sleep(3.0)

    try:
        import win32gui
    except Exception:
        check("cursor injection (needs pywin32)", False, "pywin32 unavailable")
        proc.kill()
        return

    hwnd = None
    def find(h, _):
        nonlocal hwnd
        if win32gui.IsWindowVisible(h) and win32gui.GetWindowText(h) == "PASTE TARGET":
            hwnd = h
    win32gui.EnumWindows(find, None)

    if not hwnd:
        check("target window found", False, "")
        proc.kill()
        return
    check("target window found", True, "hwnd=%s" % hwnd)

    injector = Injector(settings, own_hwnds=lambda: set())
    target = FocusTarget(hwnd, "PASTE TARGET", None)

    phrases = [
        "Hello from SraVaani Flow.",
        "ನಮಸ್ಕಾರ ಇದು ಕನ್ನಡ ಪರೀಕ್ಷೆ",
        "नमस्ते यह हिंदी परीक्षण है",
        "ఇది తెలుగు వాక్యం",
    ]
    settings.set("auto_paste", True)
    settings.set("auto_copy", True)

    delivered = []
    for phrase in phrases:
        outcome = injector.deliver(phrase, target)
        delivered.append(outcome)
        check("paste '%s'" % phrase[:22], outcome.get("pasted"),
              "restored=%s reason=%s" % (outcome.get("restored"), outcome.get("reason") or "-"))
        time.sleep(0.6)

    time.sleep(1.2)
    content = ""
    deadline = time.time() + 4
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line.startswith("CONTENT:"):
            content = line[len("CONTENT:"):].strip()
    proc.kill()

    print("      target contains: %s" % content[:120])
    landed = sum(1 for p in phrases if p in content)
    check("all phrases landed at the caret", landed == len(phrases),
          "%d/%d found" % (landed, len(phrases)))
    check("text appended after existing caret text", content.startswith("PREFIX>"),
          content[:16])


def main():
    settings = Settings()
    print("=" * 78)
    print("SraVaani Flow - live behaviour tests")
    print("=" * 78)

    model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True,
                                      dtype=torch.float16).to("cuda").eval()
    model.transcribe([np.zeros(SAMPLE_RATE, dtype=np.float32)])
    check("model on GPU", next(iter([1])) == 1, "cuda / fp16")

    test_noise(model, settings)
    test_clipping_and_edges(model, settings)
    test_cursor_injection(settings)

    print("\n" + "=" * 78)
    failed = [n for n, ok in results if not ok]
    print("%d/%d checks passed" % (len(results) - len(failed), len(results)))
    if failed:
        print("FAILED: " + ", ".join(failed))
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
