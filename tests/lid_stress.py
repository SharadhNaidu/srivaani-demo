from __future__ import annotations

import collections
import glob
import os
import sys
import unicodedata

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoModel  # noqa: E402

from sravaani_flow.decoding import ScriptMasker, greedy_decode  # noqa: E402
from sravaani_flow.config import MODEL_REPO  # noqa: E402

SR = 16000
CANDIDATES = ["LATIN", "DEVANAGARI", "KANNADA", "TELUGU", "TAMIL",
              "MALAYALAM", "BENGALI", "GUJARATI", "GURMUKHI", "ORIYA"]


def resample(x, a, b):
    if a == b:
        return x.astype(np.float32)
    n = int(round(x.size * b / a))
    X = np.fft.rfft(x)
    Y = np.zeros(n // 2 + 1, dtype=complex)
    k = min(X.size, Y.size)
    Y[:k] = X[:k]
    return (np.fft.irfft(Y, n=n) * (n / x.size)).astype(np.float32)


def load(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    return resample(w, sr, SR)


def add_noise(sig, noise, snr_db, rng):
    if noise.size < sig.size:
        noise = np.tile(noise, int(np.ceil(sig.size / noise.size)))
    noise = noise[:sig.size]
    ps, pn = np.mean(sig ** 2), np.mean(noise ** 2) + 1e-12
    return (sig + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))).astype(np.float32)


def reverb(x, sr=SR, decay=0.3, delay_ms=45):
    d = int(sr * delay_ms / 1000.0)
    out = x.copy()
    if d < x.size:
        out[d:] += x[:-d] * decay
    return (out / max(np.max(np.abs(out)), 1e-6) * 0.9).astype(np.float32)


def script_of(text):
    c = collections.Counter()
    for ch in text:
        if ch.isalpha():
            try:
                c[unicodedata.name(ch).split(" ")[0]] += 1
            except ValueError:
                pass
    return c.most_common(1)[0][0] if c else "NONE"


def encode(model, wav):
    dev = model._anchor.device
    wav_t = torch.as_tensor(np.ascontiguousarray(wav, dtype=np.float32)).reshape(-1)
    feats, flen = model.extract_features(wav_t.unsqueeze(0), torch.tensor([wav_t.shape[0]]))
    enc, enc_len = model.encoder(feats.to(model._io_dtype), flen.to(dev))
    return enc, int(enc_len[0].item())


def score_scripts(model, masker, enc, T):
    out = {}
    for script in CANDIDATES:
        _, _, st = greedy_decode(model, enc[0:1], T,
                                 mask=masker.mask_for(script), score=True)
        out[script] = st
    return out


def pick(scores, metric):
    best, best_val = None, -1e18
    for script, st in scores.items():
        if metric == "total/frames":
            v = st["total_logprob"] / max(st["frames"], 1)
        elif metric == "emitted/tokens":
            v = st["emitted_logprob"] / max(st["tokens"], 1) if st["tokens"] else -1e9
        else:
            v = st["total_logprob"] / max(st["steps"], 1)
        if v > best_val:
            best, best_val = script, v
    return best


def main():
    model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True,
                                      dtype=torch.float16).to("cuda").eval()
    model.transcribe([np.zeros(SR, dtype=np.float32)])
    sp = model._get_tokenizer()
    masker = ScriptMasker(sp, model.config.vocab_size, model.config.blank_id,
                          model._anchor.device)

    files = sorted(glob.glob(os.path.join(ROOT, "tests", "audio_lid", "*.wav")))
    rng = np.random.default_rng(3)
    metrics = ["total/frames", "total/steps", "emitted/tokens"]

    base_clips = [(os.path.basename(f), load(f)) for f in files]

    variants = []
    for name, w in base_clips:
        variants.append(("clean " + name, w))
        variants.append(("short " + name, w[:int(SR * 1.2)]))
        variants.append(("vshort " + name, w[:int(SR * 0.7)]))
        hiss = rng.standard_normal(w.size).astype(np.float32)
        variants.append(("noisy10 " + name, add_noise(w, hiss, 10, rng)))
        variants.append(("noisy5 " + name, add_noise(w, hiss, 5, rng)))
        variants.append(("reverb " + name, reverb(w)))
        variants.append(("quiet " + name, (w * 0.06).astype(np.float32)))

    auto_ok = 0
    m_ok = {m: 0 for m in metrics}
    total = 0
    failures = []

    for label, wav in variants:
        if wav.size < SR * 0.3:
            continue
        enc, T = encode(model, wav)
        toks, _ = greedy_decode(model, enc[0:1], T, mask=None)
        text = sp.decode([int(x) for x in toks])
        auto = script_of(text)

        scores = score_scripts(model, masker, enc, T)
        picks = {m: pick(scores, m) for m in metrics}

        total += 1
        if auto == "LATIN":
            auto_ok += 1
        else:
            failures.append((label, auto, text[:56], dict(picks)))
        for m in metrics:
            if picks[m] == "LATIN":
                m_ok[m] += 1

    print("=" * 76)
    print("English audio under realistic degradation; correct answer is always LATIN")
    print("=" * 76)
    if failures:
        print("\nCases where unconstrained auto-detect got it WRONG (%d):" % len(failures))
        for label, auto, text, picks in failures[:24]:
            fixed = [m for m in metrics if picks[m] == "LATIN"]
            print("  %-30s -> %-11s %s" % (label[:30], auto, text))
            print("     scored LID fixes it with: %s"
                  % (", ".join(fixed) if fixed else "NONE"))
    else:
        print("\nauto-detect never failed on these variants")

    print("\n" + "-" * 76)
    print("  unconstrained auto-detect : %d/%d  (%.1f%%)"
          % (auto_ok, total, 100.0 * auto_ok / max(total, 1)))
    for m in metrics:
        print("  scored LID [%-14s]: %d/%d  (%.1f%%)"
              % (m, m_ok[m], total, 100.0 * m_ok[m] / max(total, 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
