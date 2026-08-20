from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench_noise import (SR, build_noises, clean, mix, resample, wer)  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModel  # noqa: E402

from sravaani_flow.audio import (_auto_gain, _denoise, _highpass,  # noqa: E402
                                 estimate_snr_db, trim_to_speech)
from sravaani_flow.config import DEFAULTS, MODEL_REPO  # noqa: E402


def hum_strength(x, sr=SR):
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    total = float(np.sum(spec)) + 1e-12
    hum = 0.0
    for f0 in (50.0, 60.0):
        for k in range(1, 4):
            band = (freqs > f0 * k - 3) & (freqs < f0 * k + 3)
            hum += float(np.sum(spec[band]))
    return hum / total


def base(x):
    return _highpass(x)


def p_none(x, nref, snr):
    return x


def p_gate_always(x, nref, snr):
    return _denoise(x, nref, 0.75)


def p_gate_low(x, nref, snr):
    return _denoise(x, nref, 0.75) if snr < 12 else x


def p_gate_low_soft(x, nref, snr):
    if snr >= 12:
        return x
    strength = 0.5 if snr >= 6 else 0.8
    return _denoise(x, nref, strength)





def p_final_nonotch(x, nref, snr):
    if snr >= 22:
        return x
    strength = 0.5 if snr >= 6 else 0.8
    return _denoise(x, nref, strength)


POLICIES = {
    "none": p_none,
    "gate_always": p_gate_always,
    "final_nonotch": p_final_nonotch,
}


def main():
    import glob
    import soundfile as sf

    files = sorted(glob.glob(os.path.join(ROOT, "tests", "audio", "*.wav")))
    model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True,
                                      dtype=torch.float16).to("cuda").eval()
    model.transcribe([np.zeros(SR, dtype=np.float32)])
    vocab = DEFAULTS["vocabulary"]

    clips, refs = [], []
    for f in files:
        w, sr = sf.read(f, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        clips.append(resample(w, sr, SR))
        refs.append(open(f.replace(".wav", ".txt"), encoding="utf-8-sig").read().strip())

    rng = np.random.default_rng(11)
    noises = build_noises(max(c.size for c in clips), rng, clips)
    snrs = (20, 15, 10, 5, 0)

    totals = {k: [] for k in POLICIES}
    low_only = {k: [] for k in POLICIES}

    names = list(POLICIES)
    print("%-12s %-4s " % ("noise", "SNR") + " ".join("%-13s" % n for n in names))
    print("-" * (18 + 14 * len(names)))

    for nname, noise in noises.items():
        for snr_db in snrs:
            row = {}
            for pname, fn in POLICIES.items():
                errs = []
                for clip, ref in zip(clips, refs):
                    noisy, scaled = mix(clip, noise, snr_db)
                    y = base(noisy)
                    nref = scaled[:SR]
                    measured = estimate_snr_db(y, nref)
                    y = fn(y, nref, measured)
                    y, _ = trim_to_speech(y)
                    y = _auto_gain(y)
                    errs.append(wer(ref, clean(model.transcribe([y])[0], vocabulary=vocab)))
                row[pname] = float(np.mean(errs))
                totals[pname].append(row[pname])
                if snr_db <= 10:
                    low_only[pname].append(row[pname])
            print("%-12s %-4d " % (nname, snr_db)
                  + " ".join("%-13.3f" % row[n] for n in names))

    print("-" * (18 + 14 * len(names)))
    print("\nMEAN WER (all conditions):")
    for k in names:
        print("   %-16s %.4f" % (k, float(np.mean(totals[k]))))
    print("\nMEAN WER (SNR <= 10 dB only):")
    for k in names:
        print("   %-16s %.4f" % (k, float(np.mean(low_only[k]))))
    best = min(names, key=lambda k: np.mean(totals[k]))
    print("\nBEST OVERALL: %s" % best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
