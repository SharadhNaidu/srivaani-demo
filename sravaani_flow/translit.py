from __future__ import annotations

import re

CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r",
    "ढ़": "rh", "फ़": "f", "य़": "y",
}

INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ऑ": "o", "ऍ": "e",
}

MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॉ": "o", "ॅ": "e",
}

VIRAMA = "्"
NASALS = {"ं": "n", "ँ": "n", "ः": "h"}

DEVANAGARI_DIGITS = {"०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
                     "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"}


def devanagari_to_latin(text):
    if not text:
        return ""
    out = []
    chars = list(text)
    i = 0
    n = len(chars)
    while i < n:
        ch = chars[i]

        if ch in CONSONANTS:
            base = CONSONANTS[ch]
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:
                out.append(base)
                i += 2
                continue
            if nxt in MATRAS:
                out.append(base + MATRAS[nxt])
                i += 2
                if i < n and chars[i] in NASALS:
                    out.append(NASALS[chars[i]])
                    i += 1
                continue
            out.append(base + "a")
            i += 1
            if i < n and chars[i] in NASALS:
                out.append(NASALS[chars[i]])
                i += 1
            continue

        if ch in INDEPENDENT_VOWELS:
            out.append(INDEPENDENT_VOWELS[ch])
            i += 1
            if i < n and chars[i] in NASALS:
                out.append(NASALS[chars[i]])
                i += 1
            continue

        if ch in DEVANAGARI_DIGITS:
            out.append(DEVANAGARI_DIGITS[ch])
            i += 1
            continue

        if ch in NASALS or ch == VIRAMA or ch in MATRAS:
            i += 1
            continue

        if ch == "।":
            out.append(".")
            i += 1
            continue

        out.append(ch)
        i += 1

    text = "".join(out)
    text = re.sub(r"aa+", "aa", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


ENGLISH_FIXUPS = [
    (r"\byunivarsitee\b", "university"), (r"\byunivarsiti\b", "university"),
    (r"\byunivarsitea\b", "university"), (r"\bkaalej\b", "college"),
    (r"\bhelo\b", "hello"), (r"\bhailo\b", "hello"),
    (r"\bvaatar\b", "water"), (r"\bvatar\b", "water"),
    (r"\bkampyootar\b", "computer"), (r"\bkampootar\b", "computer"),
    (r"\bprojekt\b", "project"), (r"\bprojekta\b", "project"),
    (r"\bdemonstreshan\b", "demonstration"),
    (r"\bprezentteshan\b", "presentation"), (r"\bprezenteshan\b", "presentation"),
    (r"\bteknolojee\b", "technology"), (r"\bteknoloji\b", "technology"),
    (r"\bmodela\b", "model"), (r"\bsistam\b", "system"),
    (r"\bspeecha\b", "speech"), (r"\bteksta\b", "text"),
    (r"\bdeta\b", "data"), (r"\bthaanka\b", "thank"),
    (r"\bwelkam\b", "welcome"), (r"\bstoodenta?\b", "student"),
    (r"\bteechara?\b", "teacher"), (r"\bklaasa?\b", "class"),
]


def polish_english(text):
    out = text
    for pattern, repl in ENGLISH_FIXUPS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    out = re.sub(r"([a-z])a\b", r"\1", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def to_latin(text, script, polish=True):
    if script == "DEVANAGARI":
        out = devanagari_to_latin(text)
        return polish_english(out) if polish else out
    return text
