from __future__ import annotations

import os
import queue
import threading
import time
import traceback

import numpy as np

from .config import MODEL_REPOS, UPSTREAM_REPO, SAMPLE_RATE
from .cleanup import clean_hypothesis, word_count
from . import decoding
from .translit import to_latin
from .languages import script_for_code, AUTO

IDLE = "idle"
LOADING = "loading"
READY = "ready"
BUSY = "busy"
FAILED = "failed"


class Job:
    def __init__(self, audio, info, target=None, tag=""):
        self.audio = audio
        self.info = info or {}
        self.target = target
        self.tag = tag
        self.created = time.time()


class Result:
    def __init__(self, text, raw, job, elapsed, error=None):
        self.text = text
        self.raw = raw
        self.job = job
        self.elapsed = elapsed
        self.error = error
        self.words = word_count(text or "")

    @property
    def ok(self):
        return self.error is None and bool(self.text)


def cuda_usable():
    """True only when a GPU is genuinely usable.

    torch.cuda.is_available() can report True while device_count() is 0 --
    switchable graphics, a driver problem, or CUDA_VISIBLE_DEVICES being
    empty. Trusting it alone makes the model load onto a device that is not
    there, which fails deep inside TorchScript deserialisation.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        if torch.cuda.device_count() < 1:
            return False
        torch.cuda.get_device_name(0)
        torch.zeros(1).cuda()
        return True
    except Exception:
        return False


def resolve_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    for name in (".env", "../.env"):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if "HF_PAT" in line or "HF_TOKEN" in line:
                        value = line.split("=", 1)[-1].strip()
                        return value.strip().strip('"').strip("'")
        except Exception:
            continue
    return None


class TranscriptionEngine:
    MIN_SPEECH_RATIO = 0.06
    MIN_SECONDS = 0.25

    def __init__(self, settings, on_status=None, on_result=None):
        self.settings = settings
        self.on_status = on_status or (lambda *a, **k: None)
        self.on_result = on_result or (lambda r: None)
        self.status = IDLE
        self.detail = ""
        self.device = "cpu"
        self.precision = "fp32"
        self.model = None
        self.load_seconds = 0.0
        self.last_rtf = 0.0
        self._queue = queue.Queue()
        self._worker = None
        self._masker = None
        self._tracker = decoding.LanguageTracker()
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def _set_status(self, status, detail=""):
        self.status = status
        self.detail = detail
        try:
            self.on_status(status, detail)
        except Exception:
            pass

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="asr-worker", daemon=True)
        self._worker.start()

    def shutdown(self):
        self._stop.set()
        self._queue.put(None)

    def submit(self, audio, info, target=None, tag=""):
        self._queue.put(Job(audio, info, target, tag))

    @property
    def pending(self):
        return self._queue.qsize()

    def _pick_device(self):
        want = str(self.settings.get("device", "auto")).lower()
        if want == "cpu":
            return "cpu"
        if cuda_usable():
            return "cuda"
        if want == "cuda":
            self._set_status(LOADING, "CUDA requested but unavailable; using CPU")
        return "cpu"

    def _load(self):
        import torch
        from transformers import AutoModel

        token = resolve_token()
        if token:
            os.environ.setdefault("HF_TOKEN", token)

        device = self._pick_device()
        want_fp16 = (str(self.settings.get("precision", "fp16")).lower() == "fp16"
                     and device == "cuda")
        dtype = torch.float16 if want_fp16 else torch.float32

        self._set_status(LOADING, "Loading SraVaani-1.0 on %s" % device.upper())
        started = time.time()

        model = None
        last = None
        for repo in MODEL_REPOS:
            kwargs = dict(trust_remote_code=True, dtype=dtype)
            if token and repo == UPSTREAM_REPO:
                kwargs["token"] = token
            try:
                model = AutoModel.from_pretrained(repo, **kwargs)
                break
            except TypeError:
                kwargs.pop("dtype", None)
                kwargs["torch_dtype"] = dtype
                try:
                    model = AutoModel.from_pretrained(repo, **kwargs)
                    break
                except Exception as exc:
                    last = exc
            except Exception as exc:
                last = exc
        if model is None:
            raise last or RuntimeError("could not load the model")

        model = model.to(device).eval()

        try:
            model.transcribe([np.zeros(SAMPLE_RATE, dtype=np.float32)])
        except Exception:
            if device == "cuda":
                device = "cpu"
                model = model.float().to("cpu").eval()
                model.transcribe([np.zeros(SAMPLE_RATE, dtype=np.float32)])
                want_fp16 = False
            else:
                raise

        try:
            sp = model._get_tokenizer()
            self._masker = decoding.ScriptMasker(
                sp, model.config.vocab_size, model.config.blank_id,
                model._anchor.device)
        except Exception:
            self._masker = None

        self.model = model
        self.device = device
        self.precision = "fp16" if want_fp16 else "fp32"
        self.load_seconds = time.time() - started
        self._set_status(READY, "%s / %s" % (device.upper(), self.precision))

    def _run(self):
        try:
            self._load()
        except Exception as exc:
            self._set_status(FAILED, self._explain(exc))
            traceback.print_exc()
            return

        while not self._stop.is_set():
            job = self._queue.get()
            if job is None:
                break
            try:
                self._handle(job)
            except Exception as exc:
                traceback.print_exc()
                self._emit(Result("", "", job, 0.0, error=str(exc)))
            finally:
                if self.status != FAILED:
                    self._set_status(READY, "%s / %s" % (self.device.upper(), self.precision))

    @staticmethod
    def _explain(exc):
        text = str(exc)
        low = text.lower()
        if "401" in text or "403" in text or "gated" in low or "authoriz" in low:
            return ("Hugging Face refused the download. The model is normally "
                    "public and needs no account; check your internet connection.")
        if "connect" in low or "resolve" in low or "network" in low or "timeout" in low:
            return "Network unavailable and the model is not cached yet."
        if "out of memory" in low:
            return "GPU out of memory. Switch Compute to CPU in Settings."
        return text[:200]

    def _emit(self, result):
        try:
            self.on_result(result)
        except Exception:
            traceback.print_exc()

    def _handle(self, job):
        audio = job.audio
        seconds = audio.size / float(SAMPLE_RATE) if audio is not None else 0.0

        if audio is None or seconds < self.MIN_SECONDS:
            self._emit(Result("", "", job, 0.0, error="too_short"))
            return
        if job.info.get("speech_ratio", 1.0) < self.MIN_SPEECH_RATIO:
            self._emit(Result("", "", job, 0.0, error="no_speech"))
            return

        with self._lock:
            self._set_status(BUSY, "Transcribing %.1fs" % seconds)
            started = time.time()
            hyp = self._transcribe(audio)
            elapsed = time.time() - started

        self.last_rtf = elapsed / max(seconds, 1e-6)
        raw = getattr(hyp, "text", "") or ""

        if self._is_junk(raw, seconds):
            self._emit(Result("", raw, job, elapsed, error="no_speech"))
            return

        text = clean_hypothesis(
            hyp,
            enabled=bool(self.settings.get("cleanup", True)),
            spoken_punctuation=bool(self.settings.get("spoken_punctuation", True)),
            vocabulary=self.settings.get("vocabulary"),
            auto_punctuate=True,
        )
        self._emit(Result(text, raw, job, elapsed))

    JUNK_TOKENS = {"um", "uh", "hmm", "mm", "hm", "ah", "eh", "oh",
                   "हूँ", "हूं", "उम", "উম", "ಉಮ್", "ఉమ్", "अं", "ಅಂ"}

    @classmethod
    def _is_junk(cls, raw, seconds):
        text = (raw or "").strip()
        if not text:
            return True
        if text.lower() in cls.JUNK_TOKENS:
            return True
        if seconds >= 1.0 and len(text) <= 2:
            return True
        return False

    def _forced_script(self):
        code = str(self.settings.get("language", AUTO) or AUTO)
        if code == AUTO:
            return None
        return script_for_code(code)

    def reset_language_memory(self):
        self._tracker.reset()

    @property
    def session_script(self):
        return self._tracker.session_script

    def _transcribe(self, audio):
        import torch
        wav = np.ascontiguousarray(audio, dtype=np.float32)
        seconds = wav.size / float(SAMPLE_RATE)
        forced = self._forced_script()

        if self._masker is None:
            return self._plain_transcribe(wav)

        script = forced
        used_prior = False
        if script is None:
            prior = self._tracker.prior_for(seconds)
            if prior is not None:
                script = prior
                used_prior = True

        try:
            with torch.inference_mode():
                out = decoding.transcribe(self.model, wav, self._masker, script)
        except Exception:
            traceback.print_exc()
            return self._plain_transcribe(wav)

        text = out["text"]

        if script and not str(text).strip():
            text = self._recover_empty(wav, script)

        detected = decoding.dominant_script(text)

        if forced is None and not used_prior:
            chosen = self._tracker.observe(detected, seconds,
                                           len(out.get("tokens") or []))
            if chosen and chosen != detected:
                try:
                    with torch.inference_mode():
                        out = decoding.transcribe(self.model, wav, self._masker, chosen)
                    text = out["text"]
                    detected = decoding.dominant_script(text) or chosen
                    used_prior = True
                except Exception:
                    traceback.print_exc()

        hyp = decoding.Hypothesis(text, out["timestamp"])
        hyp.script = detected
        hyp.prior_applied = used_prior
        return hyp

    def _recover_empty(self, wav, target_script):
        """A constrained decode can come back empty on very short clips.

        Losing the utterance outright is worse than recovering it, so fall back
        to a free decode and transliterate if the script came out wrong.
        """
        import torch
        try:
            with torch.inference_mode():
                free = decoding.transcribe(self.model, wav, self._masker, None)
        except Exception:
            return ""
        text = free.get("text") or ""
        if not text.strip():
            return ""
        got = decoding.dominant_script(text)
        if got and got != target_script and target_script == "LATIN":
            return to_latin(text, got)
        return text

    def _plain_transcribe(self, wav):
        import torch
        try:
            with torch.inference_mode():
                return self.model.transcribe([wav], return_hypotheses=True, timestamps=True)[0]
        except Exception:
            with torch.inference_mode():
                out = self.model.transcribe([wav], return_hypotheses=True)
            return out[0]
