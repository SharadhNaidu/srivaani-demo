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
def greedy_decode(model, enc_out, T, mask=None):
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
            if k != blank:
                toks.append(k)
                frames.append(t)
                h, c, last = h2, c2, k
            added += 1
            t += skip
            need = (skip == 0)
        if added == cfg.max_symbols:
            t += 1
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


class Hypothesis:
    def __init__(self, text, timestamp=None):
        self.text = text
        self.timestamp = timestamp
        self.y_sequence = []
