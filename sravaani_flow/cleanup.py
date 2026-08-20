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
    return re.sub(r"(?<!\S)([^\s.,;:!?]{1,12})(\s+\1){2,}(?!\S)", r"\1", text,
                  flags=re.IGNORECASE)


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

    if enabled:
        out = _collapse_stutters(out)
    if enabled and latin:
        out = _strip_fillers(out)

    if spoken_punctuation and latin:
        out = _apply_spoken_punct(out)

    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = out.strip()

    if vocabulary:
        out = apply_vocabulary(out, vocabulary)

    if enabled and latin:
        out = merge_split_compounds(out, vocabulary)
        out = split_merged_words(out)
        out = fix_confusions(out)

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


DANDA_SCRIPTS = {"DEVANAGARI", "BENGALI", "GURMUKHI", "ORIYA", "GUJARATI"}


def sentence_mark_for(script):
    return "।" if script in DANDA_SCRIPTS else "."


def segment_by_pauses(word_ts, *, sentence_pause: float = SENTENCE_PAUSE,
                      clause_pause: float = CLAUSE_PAUSE,
                      sentence_mark: str = ".") -> str:
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
            pieces.append(sentence_mark + "\n")
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
    if auto_punctuate and words:
        try:
            text = segment_by_pauses(
                words, sentence_mark=sentence_mark_for(script_of(text)))
        except Exception:
            text = getattr(hyp, "text", text)
    return clean(text, enabled=enabled, spoken_punctuation=spoken_punctuation,
                 vocabulary=vocabulary)


COMPOUNDS = set("""
tomorrow today tonight yesterday everyone everybody everything everywhere
someone somebody something somewhere anyone anybody anything anywhere
nobody nothing nowhere myself yourself himself herself itself ourselves
themselves cannot maybe already although because before behind below
beside between beyond within without inside outside into onto upon
however therefore meanwhile otherwise nevertheless nonetheless whatever
whenever wherever whoever whichever forever forward backward afternoon
weekend birthday classroom keyboard notebook laptop software hardware
database website online offline username password filename framework
homework football basketball breakfast sunlight daylight nowadays
understand understood update upload download upgrade output input
overall overview underline background foreground feedback download
newspaper bookmark bedroom bathroom kitchen airport railway highway
worldwide lifetime sometimes anymore another herself throughout
whereas whereby herein hereby thereby therein moreover furthermore
alongside altogether anyway everyday somehow somewhat wherein
airplane aircraft spacecraft handbook workbook textbook notepad
timetable timeline deadline headline guideline baseline pipeline
network framework wallpaper screenshot smartphone microphone headphone
loudspeaker earphone playback download setup login logout signup
""".split())

_MERGE_EXCEPTIONS = {("a", "part"), ("no", "body"), ("in", "to"), ("on", "to")}


def merge_split_compounds(text, extra=None):
    if not text or not is_latin(text):
        return text
    vocab = set(COMPOUNDS)
    for word in (extra or []):
        w = str(word).strip().lower()
        if w and " " not in w:
            vocab.add(w)

    tokens = re.split(r"(\s+)", text)
    words = [(i, t) for i, t in enumerate(tokens) if t.strip()]
    out = list(tokens)
    skip = set()
    for idx in range(len(words) - 1):
        i, first = words[idx]
        j, second = words[idx + 1]
        if i in skip or j in skip:
            continue
        a = re.sub(r"[^\w']", "", first).lower()
        b = re.sub(r"[^\w']", "", second).lower()
        if not a or not b:
            continue
        if (a, b) in _MERGE_EXCEPTIONS:
            continue
        joined = a + b
        if joined in vocab and a not in vocab:
            trailing = second[len(b):] if second.lower().startswith(b) else ""
            merged = joined
            if first[:1].isupper():
                merged = merged.capitalize()
            out[i] = merged + trailing
            out[j] = ""
            for k in range(i + 1, j):
                if not tokens[k].strip():
                    out[k] = ""
            skip.add(i)
            skip.add(j)
    result = "".join(out)
    return re.sub(r"[ \t]{2,}", " ", result).strip()


SPLIT_WORDS = {
    "alot": "a lot", "infront": "in front", "eachother": "each other",
    "aswell": "as well", "atleast": "at least", "incase": "in case",
    "thankyou": "thank you", "goodmorning": "good morning",
    "goodevening": "good evening", "goodnight": "good night",
    "everytime": "every time", "inspite": "in spite", "ofcourse": "of course",
    "nomore": "no more", "somemore": "some more", "eventhough": "even though",
    "sofar": "so far", "asap": "as soon as possible",
}

CONFUSIONS = [
    (r"\bi\s+is\b", "I am"),
    (r"\bwe\s+is\b", "we are"),
    (r"\bthey\s+is\b", "they are"),
    (r"\byou\s+is\b", "you are"),
    (r"\bdont\b", "don't"), (r"\bcant\b", "can't"), (r"\bwont\b", "won't"),
    (r"\bdoesnt\b", "doesn't"), (r"\bdidnt\b", "didn't"),
    (r"\bisnt\b", "isn't"), (r"\bwasnt\b", "wasn't"), (r"\barent\b", "aren't"),
    (r"\bshouldnt\b", "shouldn't"), (r"\bcouldnt\b", "couldn't"),
    (r"\bwouldnt\b", "wouldn't"), (r"\bhavent\b", "haven't"),
    (r"\bhasnt\b", "hasn't"), (r"\bthats\b", "that's"), (r"\bits a\b", "it's a"),
    (r"\blets\b", "let's"), (r"\bim\b", "I'm"), (r"\bive\b", "I've"),
    (r"\bill\s+(be|send|do|go|call|check|make)\b", r"I'll \1"),
]


def split_merged_words(text):
    if not text or not is_latin(text):
        return text
    def repl(m):
        word = m.group(0)
        low = word.lower()
        if low not in SPLIT_WORDS:
            return word
        fixed = SPLIT_WORDS[low]
        return fixed.capitalize() if word[:1].isupper() else fixed
    pattern = r"(?<!\w)(?:" + "|".join(sorted(SPLIT_WORDS, key=len, reverse=True)) + r")(?!\w)"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def fix_confusions(text):
    if not text or not is_latin(text):
        return text
    out = text
    for pattern, repl in CONFUSIONS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out
