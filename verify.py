from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

problems = []
warnings = []


def ok(msg, detail=""):
    print("  [ok]   %-38s %s" % (msg, detail))


def warn(msg, detail=""):
    warnings.append(msg)
    print("  [warn] %-38s %s" % (msg, detail))


def fail(msg, detail=""):
    problems.append(msg)
    print("  [FAIL] %-38s %s" % (msg, detail))


def main():
    print()
    print("=" * 64)
    print(" verifying the sravaani flow install")
    print("=" * 64)

    print("\n1. python and packages")
    v = sys.version_info
    if (3, 10) <= (v.major, v.minor) < (3, 13):
        ok("python version", "%d.%d.%d" % (v.major, v.minor, v.micro))
    else:
        fail("python version", "%d.%d needs to be 3.10-3.12" % (v.major, v.minor))

    for mod in ("torch", "transformers", "sentencepiece", "numpy", "scipy",
                "soundfile", "noisereduce", "webrtcvad", "pynput", "pyperclip"):
        try:
            __import__(mod)
        except Exception as exc:
            fail("import %s" % mod, str(exc)[:44])
    if not problems:
        ok("all packages import", "")

    try:
        import win32api  # noqa: F401
        ok("pywin32 (paste at cursor)", "")
    except Exception:
        warn("pywin32 not working",
             "run .venv\\Scripts\\python.exe .venv\\Scripts\\pywin32_postinstall.py -install")

    print("\n2. compute device")
    device = "cpu"
    try:
        import torch
        from sravaani_flow.engine import cuda_usable
        if cuda_usable():
            device = "cuda"
            ok("nvidia gpu", torch.cuda.get_device_name(0))
            ok("torch build", torch.__version__)
        else:
            ok("torch build", torch.__version__ + " (running on cpu)")
            warn("no usable gpu",
                 "cpu is fine - still faster than real time")
    except Exception as exc:
        fail("torch", str(exc)[:60])

    print("\n3. model")
    model = None
    try:
        import numpy as np
        import torch
        from transformers import AutoModel
        from sravaani_flow.config import MODEL_REPOS

        started = time.time()
        last = None
        for repo in MODEL_REPOS:
            try:
                dtype = torch.float16 if device == "cuda" else torch.float32
                model = AutoModel.from_pretrained(repo, trust_remote_code=True,
                                                  dtype=dtype).to(device).eval()
                ok("model loaded", "%s in %.1fs" % (repo, time.time() - started))
                break
            except Exception as exc:
                last = exc
        if model is None:
            fail("model download", str(last)[:120])
        else:
            model.transcribe([np.random.RandomState(0).randn(48000).astype(np.float32) * 0.01])
            ok("warm-up transcribe", "")
    except Exception as exc:
        fail("model", str(exc)[:120])

    print("\n4. speech recognition")
    if model is not None:
        try:
            import soundfile as sf
            from sravaani_flow.cleanup import clean
            wav_path = os.path.join(ROOT, "sample.wav")
            if not os.path.exists(wav_path):
                warn("sample.wav missing", "skipping recognition check")
            else:
                wav, sr = sf.read(wav_path, dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(1)
                text = clean(model.transcribe([wav])[0])
                started = time.time()
                text = clean(model.transcribe([wav])[0])
                elapsed = time.time() - started
                seconds = len(wav) / float(sr)
                if "class demonstration" in text.lower():
                    ok("transcribed the sample",
                       "%.1fx real time" % (seconds / max(elapsed, 1e-6)))
                    print("         \"%s\"" % text[:56])
                else:
                    fail("transcription looks wrong", text[:60])
        except Exception as exc:
            fail("transcription", str(exc)[:120])

    print("\n5. microphone (optional for setup)")
    try:
        from sravaani_flow.audio import AudioEngine
        devices = AudioEngine.list_devices()
        if devices:
            ok("input devices found", "%d available" % len(devices))
        else:
            warn("no microphone detected",
                 "plug one in before dictating; setup is still fine")
    except Exception as exc:
        warn("could not list microphones", str(exc)[:44])

    print("\n" + "=" * 64)
    if problems:
        print(" SETUP INCOMPLETE - %d problem(s):" % len(problems))
        for p in problems:
            print("   - %s" % p)
        print("=" * 64)
        return 1

    if warnings:
        print(" READY, with %d warning(s):" % len(warnings))
        for w in warnings:
            print("   - %s" % w)
    else:
        print(" READY. everything checks out.")
    print()
    print(" launch with run.bat")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
