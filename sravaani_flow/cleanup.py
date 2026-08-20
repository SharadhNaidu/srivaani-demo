from __future__ import annotations

import re
import unicodedata

FILLERS = [
    "um", "uh", "umm", "uhh", "erm", "er", "ah", "aah",
    "hmm", "hmmm", "mmm", "mhm",
]
FILLER_PHRASES = [
    "you know what i mean", "if that makes sense", "or something like that",
    "i mean like", "sort of like", "kind of like",
    "you know", "i mean", "like i said", "basically", "actually",
    "literally", "sort of", "kind of",
]

SPOKEN_PUNCT = [
    (r"\bnew paragraph\b", "\n\n"),
    (r"\bnew line\b", "\n"),
    (r"\bnext line\b", "\n"),
    (r"\bfull stop\b", "."),
    (r"\bperiod\b", "."),
    (r"\bcomma\b", ","),
    (r"\bquestion mark\b", "?"),
    (r"\bexclamation mark\b", "!"),
    (r"\bexclamation point\b", "!"),
    (r"\bsemicolon\b", ";"),
    (r"\bcolon\b", ":"),
    (r"\bopen paren(?:thesis)?\b", "("),
    (r"\bclose paren(?:thesis)?\b", ")"),
    (r"\bhyphen\b", "-"),
    (r"\bdash\b", " - "),
]

_LATIN = re.compile(r"[A-Za-z]")


def script_of(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        script = name.split(" ")[0]
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "UNKNOWN"
    return max(counts, key=counts.get)


def is_latin(text: str) -> bool:
    return script_of(text) == "LATIN"


def _strip_fillers(text: str) -> str:
    out = text
    for phrase in FILLER_PHRASES:
        out = re.sub(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)[,]?\s*", " ", out,
                     flags=re.IGNORECASE)
    filler_re = r"(?<!\w)(?:" + "|".join(FILLERS) + r")(?!\w)[,]?\s*"
    out = re.sub(filler_re, " ", out, flags=re.IGNORECASE)
    return out


def _collapse_stutters(text: str) -> str:
    return re.sub(r"(?<!\w)(\w{1,6})(\s+\1){2,}(?!\w)", r"\1", text, flags=re.IGNORECASE)


def _apply_spoken_punct(text: str) -> str:
    out = text
    for pattern, repl in SPOKEN_PUNCT:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,.;:!?])(?=[^\s\d])", r"\1 ", out)
    out = re.sub(r"\(\s+", "(", out)
    out = re.sub(r"\s+\)", ")", out)
    return out


def _capitalise(text: str) -> str:
    def up(m):
        return m.group(0).upper()
    out = re.sub(r"(?:^|(?<=[.!?])\s+|(?<=\n))([a-z])", up, text)
    out = re.sub(r"(?<!\w)i(?!\w)", "I", out)
    return out


BUILTIN_ALIASES = {
    "SraVaani": ["sravani", "shravani", "sravaani", "shravaani", "sarvani",
                 "sra vani", "shra vani", "sri vani", "srivani",
                 "s ray vauni", "ray vauni", "sray vauni", "s ravani",
                 "shra vaani", "sarah vani"],
    "ARTPARK": ["art park", "artpark", "aart park", "art parc"],
    "IISc": ["i i s c", "iisc", "i isc", "i is c", "i i sc", "ai i s c",
             "indian institute of science"],
    "Vaani": ["vani", "vaani", "wani"],
    "FastConformer": ["fast conformer", "fastconformer"],
    "Bengaluru": ["bengaluru", "bangalore", "bengalooru"],
}


def _alias_pass(text: str, vocabulary) -> str:
    wanted = {str(w).strip().lower(): str(w).strip() for w in (vocabulary or [])}
    out = text
    for canonical, variants in BUILTIN_ALIASES.items():
        if wanted and canonical.lower() not in wanted:
            continue
        for variant in sorted(variants, key=len, reverse=True):
            pattern = r"(?<!\w)" + re.escape(variant).replace(r"\ ", r"[\s-]+") + r"(?!\w)"
            try:
                out = re.sub(pattern, canonical, out, flags=re.IGNORECASE)
            except re.error:
                continue
    return out


def apply_vocabulary(text: str, vocabulary) -> str:
    out = _alias_pass(text, vocabulary)
    for word in vocabulary or []:
        word = str(word).strip()
        if not word:
            continue
        loose = r"\s*[-\s]?\s*".join(re.escape(c) for c in word if not c.isspace())
        pattern = r"(?<!\w)" + loose + r"(?!\w)"
        try:
            out = re.sub(pattern, word, out, flags=re.IGNORECASE)
        except re.error:
            continue
    return out


def clean(text: str, *, enabled: bool = True, spoken_punctuation: bool = True,
          vocabulary=None) -> str:
    if not text:
        return ""
    out = text.strip()
    if not out:
        return ""

    latin = is_latin(out)

    if enabled and latin:
        out = _strip_fillers(out)
        out = _collapse_stutters(out)

    if spoken_punctuation and latin:
        out = _apply_spoken_punct(out)

    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = out.strip()

    if vocabulary:
        out = apply_vocabulary(out, vocabulary)

    if enabled and latin and out:
        out = _capitalise(out)

    return out


def word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


SEC_PER_CHAR = 0.075
BASE_WORD_SEC = 0.10
SENTENCE_PAUSE = 0.55
CLAUSE_PAUSE = 0.28


def _excess_silence(word: str, start: float, end: float) -> float:
    expected = BASE_WORD_SEC + SEC_PER_CHAR * max(len(word.strip()), 1)
    return max((end - start) - expected, 0.0)


def segment_by_pauses(word_ts, *, sentence_pause: float = SENTENCE_PAUSE,
                      clause_pause: float = CLAUSE_PAUSE) -> str:
    if not word_ts:
        return ""
    pieces = []
    words = [w for w in word_ts if str(w.get("word", "")).strip()]
    for i, w in enumerate(words):
        token = str(w["word"]).strip()
        pieces.append(token)
        if i == len(words) - 1:
            break
        excess = _excess_silence(token, float(w["start"]), float(w["end"]))
        if len(token) <= 2 and excess < sentence_pause:
            pieces.append(" ")
            continue
        if excess >= sentence_pause:
            pieces.append(".\n")
        elif excess >= clause_pause:
            pieces.append(", ")
        else:
            pieces.append(" ")
    text = "".join(pieces)
    text = re.sub(r"\s*\.\s*\n\s*", ".\n", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip()


def clean_hypothesis(hyp, *, enabled: bool = True, spoken_punctuation: bool = True,
                     vocabulary=None, auto_punctuate: bool = True) -> str:
    text = getattr(hyp, "text", None) or str(hyp)
    ts = getattr(hyp, "timestamp", None) or {}
    words = ts.get("word") if isinstance(ts, dict) else None
    if auto_punctuate and words and is_latin(text):
        try:
            text = segment_by_pauses(words)
        except Exception:
            text = getattr(hyp, "text", text)
    return clean(text, enabled=enabled, spoken_punctuation=spoken_punctuation,
                 vocabulary=vocabulary)
