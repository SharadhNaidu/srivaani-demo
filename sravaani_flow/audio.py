from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 480
PREROLL_SECONDS = 0.4
NOISE_PROFILE_SECONDS = 1.5


class AudioError(RuntimeError):
    pass


def _highpass(x, sr=SAMPLE_RATE, cutoff=80.0):
    try:
        from scipy.signal import butter, sosfilt
        sos = butter(4, cutoff / (sr / 2.0), btype="highpass", output="sos")
        return np.asarray(sosfilt(sos, x), dtype=np.float32)
    except Exception:
        return (x - float(np.mean(x))).astype(np.float32) if x.size else x


def _denoise(x, noise, strength, sr=SAMPLE_RATE):
    try:
        import noisereduce as nr
        kwargs = dict(y=x, sr=sr, stationary=True,
                      prop_decrease=float(np.clip(strength, 0.0, 1.0)))
        if noise is not None and noise.size >= sr // 4:
            kwargs["y_noise"] = noise
        out = np.asarray(nr.reduce_noise(**kwargs), dtype=np.float32)
        if out.size == x.size and np.isfinite(out).all():
            return out
        return x
    except Exception:
        return x


DENOISE_SNR_CEILING = 22.0


def estimate_snr_db(signal, noise):
    if signal is None or signal.size == 0:
        return 0.0
    ps = float(np.mean(signal.astype(np.float64) ** 2))
    if noise is None or noise.size == 0:
        pn = float(np.percentile(signal.astype(np.float64) ** 2, 10))
    else:
        pn = float(np.mean(noise.astype(np.float64) ** 2))
    if pn <= 1e-12 or ps <= 1e-12:
        return 60.0
    return float(10.0 * np.log10(ps / pn))


def _auto_gain(x, target_peak=0.92, max_gain=12.0):
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak < 1e-4:
        return x
    gain = min(target_peak / peak, max_gain)
    return np.clip(x * gain, -1.0, 1.0).astype(np.float32)


def vad_flags(x, sr=SAMPLE_RATE, aggressiveness=2):
    try:
        import webrtcvad
    except Exception:
        return None
    try:
        vad = webrtcvad.Vad(int(np.clip(aggressiveness, 0, 3)))
        pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        frame_bytes = BLOCK * 2
        flags = []
        for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
            flags.append(vad.is_speech(pcm[i:i + frame_bytes], sr))
        return flags or None
    except Exception:
        return None


def trim_to_speech(x, sr=SAMPLE_RATE, pad_ms=200):
    if x.size == 0:
        return x, 0.0

    flags = vad_flags(x, sr)
    if not flags:
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        return x, (1.0 if rms > 0.004 else 0.0)

    speech_ratio = sum(flags) / float(len(flags))
    if not any(flags):
        return x, 0.0
    first = flags.index(True)
    last = len(flags) - 1 - flags[::-1].index(True)
    pad = int(sr * pad_ms / 1000.0)
    start = max(0, first * BLOCK - pad)
    end = min(int(x.size), (last + 1) * BLOCK + pad)
    return x[start:end], speech_ratio


class AudioEngine:

    def __init__(self, settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._stream = None
        self._recording = False
        self._frames = []
        self._preroll = deque(maxlen=max(1, int(PREROLL_SECONDS * SAMPLE_RATE / BLOCK)))
        self._noise = deque(maxlen=max(1, int(NOISE_PROFILE_SECONDS * SAMPLE_RATE / BLOCK)))
        self._level = 0.0
        self._started_at = 0.0
        self.overflow_count = 0
        self.waveform = deque([0.0] * 72, maxlen=72)
        self.last_error = None

    @staticmethod
    def list_devices():
        try:
            import sounddevice as sd
            out = []
            for idx, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    out.append((idx, str(d.get("name", "Device %d" % idx))))
            return out
        except Exception:
            return []

    def start(self):
        import sounddevice as sd
        self.stop()
        device = self.settings.get("input_device")
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=BLOCK, device=device, callback=self._callback,
                latency="low",
            )
            self._stream.start()
            self.last_error = None
        except Exception as exc:
            self._stream = None
            self.last_error = str(exc)
            raise AudioError("Could not open microphone: %s" % exc) from exc

    def stop(self):
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    @property
    def running(self):
        return self._stream is not None and getattr(self._stream, "active", False)

    def _callback(self, indata, frames, time_info, status):
        try:
            if status:
                self.overflow_count += 1
            block = np.asarray(indata[:, 0], dtype=np.float32).copy()

            rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)) + 1e-12)
            alpha = 0.5 if rms > self._level else 0.15
            self._level = (1 - alpha) * self._level + alpha * rms

            with self._lock:
                self.waveform.append(min(rms * 12.0, 1.0))
                if self._recording:
                    self._frames.append(block)
                else:
                    self._preroll.append(block)
                    if rms < 0.02:
                        self._noise.append(block)
        except Exception:
            pass

    def begin(self):
        with self._lock:
            self._frames = list(self._preroll)
            self._recording = True
            self._started_at = time.time()

    @property
    def is_recording(self):
        return self._recording

    @property
    def elapsed(self):
        return (time.time() - self._started_at) if self._recording else 0.0

    @property
    def level(self):
        return self._level

    def noise_profile(self):
        with self._lock:
            if not self._noise:
                return None
            try:
                return np.concatenate(list(self._noise))
            except Exception:
                return None

    def end(self):
        with self._lock:
            self._recording = False
            frames, self._frames = self._frames, []
        if not frames:
            return np.zeros(0, dtype=np.float32)
        try:
            return np.concatenate(frames).astype(np.float32)
        except Exception:
            return np.zeros(0, dtype=np.float32)

    def cancel(self):
        with self._lock:
            self._recording = False
            self._frames = []

    def process(self, audio):
        info = {"raw_seconds": audio.size / float(SAMPLE_RATE), "speech_ratio": 1.0,
                "denoised": False, "trimmed": False,
                "seconds": audio.size / float(SAMPLE_RATE)}
        if audio.size == 0:
            info["speech_ratio"] = 0.0
            return audio, info

        if self.settings.get("highpass", True):
            audio = _highpass(audio)

        noise_ref = self.noise_profile()
        info["snr_db"] = estimate_snr_db(audio, noise_ref)
        if self.settings.get("denoise", True) and info["snr_db"] < DENOISE_SNR_CEILING:
            before = audio
            audio = _denoise(audio, noise_ref,
                             float(self.settings.get("denoise_strength", 0.75)))
            info["denoised"] = audio is not before

        trimmed, ratio = trim_to_speech(audio)
        info["speech_ratio"] = ratio
        if self.settings.get("vad_trim", True):
            if trimmed.size >= SAMPLE_RATE * 0.15:
                audio = trimmed
                info["trimmed"] = True

        if self.settings.get("auto_gain", True):
            audio = _auto_gain(audio)

        info["seconds"] = audio.size / float(SAMPLE_RATE)
        return audio, info
