from __future__ import annotations

import collections
import unicodedata

import numpy as np
import torch

NEUTRAL = "NEUTRAL"

SCRIPT_COMPANIONS = {
    "LATIN": {"LATIN"},
    "DEVANAGARI": {"DEVANAGARI"},
    "BENGALI": {"BENGALI"},
    "KANNADA": {"KANNADA"},
    "TELUGU": {"TELUGU"},
    "TAMIL": {"TAMIL"},
    "MALAYALAM": {"MALAYALAM"},
    "GUJARATI": {"GUJARATI"},
    "GURMUKHI": {"GURMUKHI"},
    "ORIYA": {"ORIYA"},
    "OL": {"OL"},
    "MEETEI": {"MEETEI", "BENGALI"},
}


def piece_script(piece):
    counts = collections.Counter()
    for ch in piece:
        if not ch.isalpha():
            continue
        try:
            counts[unicodedata.name(ch).split(" ")[0]] += 1
        except ValueError:
            continue
    if not counts:
        return NEUTRAL
    return counts.most_common(1)[0][0]


class ScriptMasker:
    def __init__(self, sp, vocab_size, blank_id, device):
        self.sp = sp
        self.vocab_size = int(vocab_size)
        self.blank_id = int(blank_id)
        self.device = device
        self._scripts = [piece_script(sp.id_to_piece(i)) for i in range(self.vocab_size)]
        self._cache = {}

    def available_scripts(self):
        return set(self._scripts) - {NEUTRAL}

    def mask_for(self, script):
        if script is None:
            return None
        if script in self._cache:
            return self._cache[script]
        allowed = SCRIPT_COMPANIONS.get(script, {script})
        size = self.blank_id + 1
        mask = torch.zeros(size, dtype=torch.bool, device=self.device)
        for i, s in enumerate(self._scripts):
            if s == NEUTRAL or s in allowed:
                mask[i] = True
        mask[self.blank_id] = True
        self._cache[script] = mask
        return mask


@torch.no_grad()
def greedy_decode(model, enc_out, T, mask=None, score=False):
    cfg = model.config
    dev = model._anchor.device
    nd = cfg.num_durations
    blank = cfg.blank_id
    durs = cfg.durations
    dtype = model._io_dtype

    h = torch.zeros(cfg.pred_rnn_layers, 1, cfg.pred_hidden, device=dev, dtype=dtype)
    c = torch.zeros(cfg.pred_rnn_layers, 1, cfg.pred_hidden, device=dev, dtype=dtype)
    last, toks, frames = blank, [], []
    tlen = torch.ones(1, dtype=torch.int32, device=dev)
    total_logprob = 0.0
    emitted_logprob = 0.0
    steps = 0
    t = 0
    while t < T:
        f = enc_out.narrow(2, t, 1)
        added, need = 0, True
        while need and added < cfg.max_symbols:
            tgt = torch.tensor([[last]], dtype=torch.int32, device=dev)
            logits, _, h2, c2 = model.decoder_joint(f, tgt, tlen, h, c)
            logits = logits[0, 0, 0]
            token_logits = logits[:-nd]
            if mask is not None:
                token_logits = token_logits.masked_fill(~mask, float("-inf"))
            k = int(token_logits.argmax().item())
            skip = durs[int(logits[-nd:].argmax().item())]
            if score:
                lp = torch.log_softmax(logits[:-nd].float(), dim=-1)[k]
                total_logprob += float(lp)
                steps += 1
                if k != blank:
                    emitted_logprob += float(lp)
            if k != blank:
                toks.append(k)
                frames.append(t)
                h, c, last = h2, c2, k
            added += 1
            t += skip
            need = (skip == 0)
        if added == cfg.max_symbols:
            t += 1
    if score:
        return toks, frames, {"total_logprob": total_logprob,
                              "emitted_logprob": emitted_logprob,
                              "steps": steps, "frames": T,
                              "tokens": len(toks)}
    return toks, frames


@torch.no_grad()
def transcribe(model, wav, masker=None, script=None, timestamps=True):
    dev = model._anchor.device
    model._ensure_loaded()
    sp = model._get_tokenizer()

    wav_t = torch.as_tensor(np.ascontiguousarray(wav, dtype=np.float32)).reshape(-1)
    feats, flen = model.extract_features(wav_t.unsqueeze(0),
                                         torch.tensor([wav_t.shape[0]]))
    enc, enc_len = model.encoder(feats.to(model._io_dtype), flen.to(dev))
    T = int(enc_len[0].item())

    mask = masker.mask_for(script) if (masker is not None and script) else None
    toks, frames = greedy_decode(model, enc[0:1], T, mask=mask)

    text = sp.decode([int(x) for x in toks])
    result = {"text": text, "tokens": toks, "frames": frames, "timestamp": None}

    if timestamps and toks:
        sub = int(getattr(model.config, "subsampling_factor", 8))
        fd = model._frame_seconds()
        total = -(-int(flen[0].item()) // sub)
        try:
            result["timestamp"] = model._make_timestamps(sp, toks, frames, fd, total)
        except Exception:
            result["timestamp"] = None
    return result


SHORT_UTTERANCE_SECONDS = 4.0
MIN_TOKENS_FOR_TRUST = 3


def dominant_script(text):
    counts = collections.Counter()
    for ch in text or "":
        if not ch.isalpha():
            continue
        try:
            counts[unicodedata.name(ch).split(" ")[0]] += 1
        except ValueError:
            continue
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def score_candidates(model, masker, enc, T, candidates):
    """Path log-likelihood per decoding step for each candidate script.

    Candidates that emit no tokens at all are excluded: an empty path scores
    deceptively well because there is nothing to be wrong about.
    """
    out = {}
    for script in candidates:
        _, _, st = greedy_decode(model, enc[0:1], T,
                                 mask=masker.mask_for(script), score=True)
        if st["tokens"] < 1:
            continue
        out[script] = st["total_logprob"] / max(st["steps"], 1)
    return out


class LanguageTracker:
    """Remembers what language the session has been in.

    Short utterances are genuinely ambiguous -- "hello" is a real Hindi
    loanword (हेलो), and the model prefers Devanagari for it by a wide margin
    (-0.10 vs -0.52 log-prob per step). No likelihood threshold separates that
    from real Hindi. What does separate them is context: if the dictation so
    far has been English, a short utterance is almost certainly still English.

    So the session language is sticky. Short utterances simply inherit it.
    Long utterances may change it, but only after agreeing twice in a row, so
    one odd result cannot flip the whole session.
    """

    def __init__(self, switch_evidence=2):
        self.switch_evidence = switch_evidence
        self._session = None
        self._pending = None
        self._pending_hits = 0

    def reset(self):
        self._session = None
        self._pending = None
        self._pending_hits = 0

    @property
    def session_script(self):
        return self._session

    def prior_for(self, seconds):
        if seconds >= SHORT_UTTERANCE_SECONDS:
            return None
        return self._session

    def observe(self, script, seconds, tokens):
        """Feed a freely-decoded result in. Returns the script to actually use."""
        if not script or tokens < MIN_TOKENS_FOR_TRUST:
            return self._session

        if self._session is None:
            self._session = script
            self._pending = None
            self._pending_hits = 0
            return script

        if script == self._session:
            self._pending = None
            self._pending_hits = 0
            return script

        if self._pending == script:
            self._pending_hits += 1
        else:
            self._pending = script
            self._pending_hits = 1

        if self._pending_hits >= self.switch_evidence:
            self._session = script
            self._pending = None
            self._pending_hits = 0
            return script

        return self._session


class Hypothesis:
    def __init__(self, text, timestamp=None):
        self.text = text
        self.timestamp = timestamp
        self.y_sequence = []
