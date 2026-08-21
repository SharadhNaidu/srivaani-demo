from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from transformers import AutoModel

from sravaani_flow.audio import _auto_gain, _denoise, _highpass, trim_to_speech
from sravaani_flow.cleanup import clean
from sravaani_flow.config import DEFAULTS, MODEL_REPO

SR = 16000


def norm(s):
    return re.sub(r"[^a-z ]", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1,
                          d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[len(r), len(h)] / max(len(r), 1)


def resample(x, a, b):
    n = int(round(x.size * b / a))
    X = np.fft.rfft(x)
    Y = np.zeros(n // 2 + 1, dtype=complex)
    k = min(X.size, Y.size)
    Y[:k] = X[:k]
    return (np.fft.irfft(Y, n=n) * (n / x.size)).astype(np.float32)


def mix(sig, noise, snr_db):
    if noise.size < sig.size:
        noise = np.tile(noise, int(np.ceil(sig.size / noise.size)))
    noise = noise[:sig.size]
    ps, pn = np.mean(sig ** 2), np.mean(noise ** 2) + 1e-12
    return (sig + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))).astype(np.float32), \
           (noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))).astype(np.float32)


def build_noises(n, rng, clips):
    t = np.arange(n) / float(SR)
    white = rng.standard_normal(n).astype(np.float32)
    hum = (np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*100*t)
           + 0.25*np.sin(2*np.pi*150*t) + 0.2*rng.standard_normal(n)).astype(np.float32)
    fan = np.convolve(rng.standard_normal(n), np.ones(96)/96.0, mode="same").astype(np.float32)

    babble = np.zeros(n, dtype=np.float32)
    for c in clips[:4]:
        off = rng.integers(0, max(len(c) - 1, 1))
        seg = np.roll(c, off)
        seg = np.tile(seg, int(np.ceil(n / seg.size)))[:n]
        babble += seg.astype(np.float32)
    if np.max(np.abs(babble)) > 0:
        babble /= np.max(np.abs(babble))

    impulse = rng.standard_normal(n).astype(np.float32) * 0.02
    for pos in rng.integers(0, n - 800, size=6):
        impulse[pos:pos+400] += (rng.standard_normal(400) * 3.0).astype(np.float32)

    keyboard = np.zeros(n, dtype=np.float32)
    for pos in rng.integers(0, n - 300, size=40):
        keyboard[pos:pos+120] += (rng.standard_normal(120)
                                  * np.hanning(120) * 1.5).astype(np.float32)

    return {"white hiss": white, "mains hum": hum, "fan rumble": fan,
            "babble": babble, "impulses": impulse, "keyboard": keyboard}


def chain_current(x, noise_ref):
    y = _highpass(x)
    y = _denoise(y, noise_ref, 0.75)
    y, _ = trim_to_speech(y)
    return _auto_gain(y)



def chain_none(x, noise_ref):
    y, _ = trim_to_speech(_highpass(x))
    return _auto_gain(y)


def main():
    wer_dir = os.path.join(ROOT, "tests", "audio")
    files = sorted(glob.glob(os.path.join(wer_dir, "*.wav")))
    if not files:
        print("no test audio found in tests/audio")
        return 1

    model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True,
                                      dtype=torch.float16).to("cuda").eval()
    model.transcribe([np.zeros(SR, dtype=np.float32)])
    vocab = DEFAULTS["vocabulary"]

    clips, refs = [], []
    for f in files:
        w, sr = __import__("soundfile").read(f, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        clips.append(resample(w, sr, SR))
        refs.append(open(f.replace(".wav", ".txt"), encoding="utf-8-sig").read().strip())

    rng = np.random.default_rng(11)
    chains = {"none": chain_none, "current": chain_current}
    snrs = (20, 15, 10, 5, 0)

    totals = {k: [] for k in chains}
    print("%-12s %-5s %-8s %-8s" % ("noise", "SNR", "none", "current"))
    print("-" * 48)

    noises = build_noises(max(c.size for c in clips), rng, clips)
    for nname, noise in noises.items():
        for snr in snrs:
            row = {}
            for cname, fn in chains.items():
                errs = []
                for clip, ref in zip(clips, refs):
                    noisy, scaled = mix(clip, noise, snr)
                    ref_noise = scaled[:SR]
                    text = model.transcribe([fn(noisy, ref_noise)])[0]
                    errs.append(wer(ref, clean(text, vocabulary=vocab)))
                row[cname] = float(np.mean(errs))
                totals[cname].append(row[cname])
            best = min(row, key=row.get)
            print("%-12s %-5d %-8.3f %-8.3f  best=%s" % (
                nname, snr, row["none"], row["current"], best))

    print("-" * 48)
    for k in chains:
        print("MEAN WER  %-8s = %.4f" % (k, float(np.mean(totals[k]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
