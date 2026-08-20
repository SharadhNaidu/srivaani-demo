from __future__ import annotations

import unicodedata

AUTO = "auto"

LANGUAGES = [
    ("auto", "Auto-detect", None, True),
    ("as", "Assamese", "BENGALI", True),
    ("bn", "Bengali", "BENGALI", True),
    ("brx", "Bodo", "DEVANAGARI", True),
    ("doi", "Dogri", "DEVANAGARI", True),
    ("en", "English", "LATIN", True),
    ("gu", "Gujarati", "GUJARATI", True),
    ("hi", "Hindi", "DEVANAGARI", True),
    ("kn", "Kannada", "KANNADA", True),
    ("kok", "Konkani", "DEVANAGARI", True),
    ("mai", "Maithili", "DEVANAGARI", True),
    ("ml", "Malayalam", "MALAYALAM", True),
    ("mni", "Manipuri", "BENGALI", True),
    ("mr", "Marathi", "DEVANAGARI", True),
    ("ne", "Nepali", "DEVANAGARI", True),
    ("or", "Odia", "ORIYA", True),
    ("pa", "Punjabi", "GURMUKHI", True),
    ("sa", "Sanskrit", "DEVANAGARI", True),
    ("sat", "Santali", "OL", True),
    ("sd", "Sindhi", "DEVANAGARI", True),
    ("ta", "Tamil", "TAMIL", True),
    ("te", "Telugu", "TELUGU", True),
    ("njm", "Angami", "LATIN", False),
    ("njo", "Ao", "LATIN", False),
    ("awa", "Awadhi", "DEVANAGARI", False),
    ("bjj", "Bajjika", "DEVANAGARI", False),
    ("bry", "Bearybashe", "KANNADA", False),
    ("bhb", "Bhili", "DEVANAGARI", False),
    ("bho", "Bhojpuri", "DEVANAGARI", False),
    ("bns", "Bundeli", "DEVANAGARI", False),
    ("nbc", "Chakhesang", "LATIN", False),
    ("ccp", "Chakma", "BENGALI", False),
    ("hne", "Chhattisgarhi", "DEVANAGARI", False),
    ("grt", "Garo", "LATIN", False),
    ("gbm", "Garhwali", "DEVANAGARI", False),
    ("gon", "Gondi", "DEVANAGARI", False),
    ("hlb", "Halbi", "DEVANAGARI", False),
    ("bgc", "Haryanvi", "DEVANAGARI", False),
    ("clk", "Idu Mishmi", "LATIN", False),
    ("mjw", "Karbi", "LATIN", False),
    ("kfx", "Khariboli", "DEVANAGARI", False),
    ("kho", "Khortha", "DEVANAGARI", False),
    ("trp", "Kokborok", "BENGALI", False),
    ("kru", "Kurukh", "DEVANAGARI", False),
    ("mag", "Magadhi", "DEVANAGARI", False),
    ("mvi", "Malvani", "DEVANAGARI", False),
    ("mwr", "Marwari", "DEVANAGARI", False),
    ("lus", "Mizo", "LATIN", False),
    ("nag", "Nagamese", "LATIN", False),
    ("njz", "Nyishi", "LATIN", False),
    ("raj", "Rajasthani", "DEVANAGARI", False),
    ("nnl", "Rengma", "LATIN", False),
    ("nbu", "Rongmei", "LATIN", False),
    ("sck", "Sadri", "DEVANAGARI", False),
    ("spv", "Sambalpuri", "ORIYA", False),
    ("nsm", "Sumi", "LATIN", False),
    ("sgj", "Surgujia", "DEVANAGARI", False),
    ("sjp", "Surjapuri", "DEVANAGARI", False),
    ("tgj", "Tagin", "LATIN", False),
    ("tcy", "Tulu", "KANNADA", False),
    ("wnc", "Wancho", "LATIN", False),
]

BY_CODE = {code: (name, script, scheduled) for code, name, script, scheduled in LANGUAGES}
BY_NAME = {name: code for code, name, _s, _sc in LANGUAGES}

SCRIPT_LABELS = {
    "DEVANAGARI": "Devanagari",
    "BENGALI": "Bengali",
    "KANNADA": "Kannada",
    "TELUGU": "Telugu",
    "TAMIL": "Tamil",
    "MALAYALAM": "Malayalam",
    "GUJARATI": "Gujarati",
    "GURMUKHI": "Gurmukhi",
    "ORIYA": "Odia",
    "LATIN": "Latin",
    "OL": "Ol Chiki",
}

SCRIPT_DEFAULT_LANGUAGE = {
    "DEVANAGARI": "hi",
    "BENGALI": "bn",
    "KANNADA": "kn",
    "TELUGU": "te",
    "TAMIL": "ta",
    "MALAYALAM": "ml",
    "GUJARATI": "gu",
    "GURMUKHI": "pa",
    "ORIYA": "or",
    "LATIN": "en",
    "OL": "sat",
}


def display_names(scheduled_first=True):
    scheduled = [n for c, n, _s, sc in LANGUAGES if sc and c != AUTO]
    other = [n for c, n, _s, sc in LANGUAGES if not sc]
    if scheduled_first:
        return ["Auto-detect"] + sorted(scheduled) + sorted(other)
    return ["Auto-detect"] + sorted(scheduled + other)


def code_for_name(name):
    return BY_NAME.get(name, AUTO)


def name_for_code(code):
    entry = BY_CODE.get(code)
    return entry[0] if entry else "Auto-detect"


def script_for_code(code):
    entry = BY_CODE.get(code)
    return entry[1] if entry else None


def detect_script(text):
    counts = {}
    for ch in text or "":
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        script = name.split(" ")[0]
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def script_label(script):
    return SCRIPT_LABELS.get(script, (script or "unknown").title())


def describe(text, selected_code=AUTO):
    script = detect_script(text)
    result = {"script": script, "script_label": script_label(script),
              "language": None, "mismatch": False, "note": ""}
    if script is None:
        return result

    if selected_code and selected_code != AUTO:
        expected = script_for_code(selected_code)
        chosen_name = name_for_code(selected_code)
        if expected and expected == script:
            result["language"] = chosen_name
        elif expected:
            result["mismatch"] = True
            result["language"] = name_for_code(SCRIPT_DEFAULT_LANGUAGE.get(script, AUTO))
            result["note"] = "expected %s, detected %s script" % (
                chosen_name, script_label(script))
        else:
            result["language"] = chosen_name
        return result

    result["language"] = name_for_code(SCRIPT_DEFAULT_LANGUAGE.get(script, AUTO))
    if script == "DEVANAGARI":
        result["note"] = "Devanagari is shared by Hindi, Marathi, Nepali and others"
    elif script == "BENGALI":
        result["note"] = "Bengali script is shared by Bengali, Assamese and Manipuri"
    elif script == "KANNADA":
        result["note"] = "Kannada script is shared by Kannada and Tulu"
    return result
