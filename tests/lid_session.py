from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from transformers import AutoModel  # noqa: E402

from lid_stress import SR, add_noise, encode, load, reverb, script_of  # noqa: E402
from sravaani_flow import decoding  # noqa: E402
from sravaani_flow.config import MODEL_REPO  # noqa: E402


def run(model, masker, tracker, wav, use_tracker):
    seconds = wav.size / float(SR)
    script = tracker.prior_for(seconds) if use_tracker else None
    used_prior = script is not None
    out = decoding.transcribe(model, wav, masker, script)
    if script and not out["text"].strip():
        free = decoding.transcribe(model, wav, masker, None)
        got = decoding.dominant_script(free["text"])
        if got and got != script and script == "LATIN":
            from sravaani_flow.translit import to_latin
            out = {"text": to_latin(free["text"], got), "tokens": free["tokens"],
                   "timestamp": free["timestamp"]}
        else:
            out = free
    detected = decoding.dominant_script(out["text"])
    if use_tracker and not used_prior:
        chosen = tracker.observe(detected, seconds, len(out["tokens"]))
        if chosen and chosen != detected:
            out = decoding.transcribe(model, wav, masker, chosen)
            detected = decoding.dominant_script(out["text"]) or chosen
            used_prior = True
    return out["text"], detected, used_prior


def main():
    model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True,
                                      dtype=torch.float16).to("cuda").eval()
    model.transcribe([np.zeros(SR, dtype=np.float32)])
    sp = model._get_tokenizer()
    masker = decoding.ScriptMasker(sp, model.config.vocab_size,
                                   model.config.blank_id, model._anchor.device)

    files = sorted(glob.glob(os.path.join(ROOT, "tests", "audio_lid", "*.wav")))
    clips = [(os.path.basename(f), load(f)) for f in files]
    rng = np.random.default_rng(3)

    long_clips = [(n, w) for n, w in clips if w.size / SR >= 2.5]
    hard = []
    for n, w in clips:
        hard.append(("short " + n, w[:int(SR * 1.2)]))
        hard.append(("vshort " + n, w[:int(SR * 0.7)]))
        hiss = rng.standard_normal(w.size).astype(np.float32)
        hard.append(("noisy10 " + n, add_noise(w, hiss, 10, rng)))
        hard.append(("reverb " + n, reverb(w)))
    hard = [(n, w) for n, w in hard if w.size >= SR * 0.3]

    print("=" * 74)
    print("Simulating a real session: the user dictates a few full English")
    print("sentences, then short/degraded utterances. Correct answer: LATIN.")
    print("=" * 74)

    for use_tracker in (False, True):
        tracker = decoding.LanguageTracker()
        label = "WITH session memory" if use_tracker else "WITHOUT session memory"

        if use_tracker:
            for n, w in long_clips[:4]:
                run(model, masker, tracker, w, True)
            print("\n%s  (session locked to %s)" % (label, tracker.session_script))
        else:
            print("\n%s" % label)

        ok = 0
        for n, w in hard:
            text, detected, prior = run(model, masker, tracker, w, use_tracker)
            good = (detected == "LATIN")
            ok += good
            if not good:
                print("   FAIL %-28s -> %-11s %s" % (n[:28], detected, text[:40]))
        print("   %d/%d correct (%.1f%%)" % (ok, len(hard), 100.0 * ok / len(hard)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
