from __future__ import annotations

import glob
import os
import sys
import time

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


def encode(model, wav):
    dev = model._anchor.device
    wav_t = torch.as_tensor(np.ascontiguousarray(wav, dtype=np.float32)).reshape(-1)
    feats, flen = model.extract_features(wav_t.unsqueeze(0), torch.tensor([wav_t.shape[0]]))
    enc, enc_len = model.encoder(feats.to(model._io_dtype), flen.to(dev))
    return enc, int(enc_len[0].item())


def main():
    model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True,
                                      dtype=torch.float16).to("cuda").eval()
    model.transcribe([np.zeros(SR, dtype=np.float32)])
    sp = model._get_tokenizer()
    masker = ScriptMasker(sp, model.config.vocab_size, model.config.blank_id,
                          model._anchor.device)

    files = sorted(glob.glob(os.path.join(ROOT, "tests", "audio_lid", "*.wav")))
    files += sorted(glob.glob(os.path.join(ROOT, "tests", "audio", "*.wav")))
    if not files:
        print("no audio")
        return 1

    metrics = ["total/frames", "total/steps", "emitted/tokens", "total_raw"]
    wins = {m: 0 for m in metrics}
    unconstrained_ok = 0
    total = 0
    timings = []

    for f in files:
        wav = load(f)
        enc, T = encode(model, wav)

        toks, _ = greedy_decode(model, enc[0:1], T, mask=None)
        auto_text = sp.decode([int(x) for x in toks])
        from sravaani_flow.decoding import piece_script
        import collections
        c = collections.Counter()
        for ch in auto_text:
            if ch.isalpha():
                try:
                    import unicodedata
                    c[unicodedata.name(ch).split(" ")[0]] += 1
                except ValueError:
                    pass
        auto_script = c.most_common(1)[0][0] if c else "NONE"

        t0 = time.time()
        scores = {}
        for script in CANDIDATES:
            _, _, st = greedy_decode(model, enc[0:1], T,
                                     mask=masker.mask_for(script), score=True)
            scores[script] = st
        elapsed = time.time() - t0
        timings.append(elapsed)

        picks = {}
        for m in metrics:
            best, best_val = None, -1e18
            for script, st in scores.items():
                if m == "total/frames":
                    v = st["total_logprob"] / max(st["frames"], 1)
                elif m == "total/steps":
                    v = st["total_logprob"] / max(st["steps"], 1)
                elif m == "emitted/tokens":
                    v = st["emitted_logprob"] / max(st["tokens"], 1) if st["tokens"] else -1e9
                else:
                    v = st["total_logprob"]
                if v > best_val:
                    best, best_val = script, v
            picks[m] = best

        total += 1
        if auto_script == "LATIN":
            unconstrained_ok += 1
        for m in metrics:
            if picks[m] == "LATIN":
                wins[m] += 1

        name = os.path.basename(f)
        flag = "OK " if auto_script == "LATIN" else "BAD"
        print("%-22s auto=%-11s %s | %s" % (
            name, auto_script, flag,
            "  ".join("%s=%s" % (m.split("/")[0][:4], picks[m][:4]) for m in metrics)))
        if auto_script != "LATIN":
            print("     auto text: %s" % auto_text[:70])

    print("\n" + "=" * 70)
    print("all clips are English; correct answer is LATIN for every one")
    print("  unconstrained auto-detect : %d/%d correct" % (unconstrained_ok, total))
    for m in metrics:
        print("  scored LID [%-14s]: %d/%d correct" % (m, wins[m], total))
    print("  scoring cost: %.2fs mean over %d scripts" % (float(np.mean(timings)), len(CANDIDATES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
