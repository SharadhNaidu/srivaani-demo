from __future__ import annotations

import json
import os
import threading
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = APP_DIR / "settings.json"
HISTORY_PATH = APP_DIR / "history.jsonl"

MODEL_REPO = "ARTPARK-IISc/SraVaani-1.0"
SAMPLE_RATE = 16000

DEFAULTS = {
    "hotkey_ptt": "shift_r",
    "hotkey_toggle": "f9",
    "hotkey_paste_last": "f11",
    "input_device": None,
    "denoise": True,
    "denoise_strength": 0.75,
    "highpass": True,
    "vad_trim": True,
    "auto_gain": True,
    "auto_paste": True,
    "auto_copy": True,
    "cleanup": True,
    "spoken_punctuation": True,
    "language": "auto",
    "device": "auto",
    "precision": "fp16",
    "min_record_seconds": 0.35,
    "max_record_seconds": 120.0,
    "vocabulary": [
        "SraVaani", "ARTPARK", "IISc", "Vaani", "FastConformer",
        "Bengaluru", "Kannada", "ASR", "TDT", "CTC",
    ],
}


class Settings:

    def __init__(self, path: Path = SETTINGS_PATH):
        self._path = path
        self._lock = threading.RLock()
        self._data = dict(DEFAULTS)
        self.load()

    def load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    with self._lock:
                        for k, v in raw.items():
                            if k in DEFAULTS:
                                self._data[k] = v
        except Exception:
            pass

    def save(self) -> None:
        try:
            tmp = self._path.with_suffix(".json.tmp")
            with self._lock:
                payload = json.dumps(self._data, indent=2, ensure_ascii=False)
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception:
            pass

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, DEFAULTS.get(key, default))

    def set(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
        self.save()

    def as_dict(self) -> dict:
        with self._lock:
            return dict(self._data)
