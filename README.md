# SraVaani Flow

Offline push-to-talk dictation on the [ARTPARK-IISc SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0)
speech recognition model. Hold a key, speak, release — the text lands at your
cursor in whatever app you were typing in.

Runs entirely on your machine. No API keys at run time, no network calls after
the first model download.

---

## Running it on another laptop with an NVIDIA GPU

### What you need first

| Requirement | How to check | If it's missing |
|---|---|---|
| Windows 10 or 11 | — | — |
| NVIDIA GPU + driver | Run `nvidia-smi` in a terminal | Install the driver from nvidia.com |
| Python 3.10, 3.11 or 3.12 | Run `py -0p` | Install from python.org. **Not 3.13+** — PyTorch has no wheels for it yet |
| ~4 GB free disk | — | Model is 900 MB, PyTorch CUDA is ~2.5 GB |
| A Hugging Face account | — | Needed once, to download the model |

### Step 1 — get access to the model

The model is gated. Open <https://huggingface.co/ARTPARK-IISc/SraVaani-1.0>,
sign in, and accept the terms (it approves automatically).

Then create a token at <https://huggingface.co/settings/tokens> with **Read**
access, and put it in a file called `.env` next to `setup.bat`:

```
HF_PAT = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

> Keep `.env` private — it is a personal credential. `.gitignore` already
> excludes it, so it will not be committed.

### Step 2 — run setup

Double-click **`setup.bat`**, or run it from a terminal.

It creates a virtual environment, installs the CUDA build of PyTorch,
downloads the model, and finishes by running the self-test.

This takes 5–15 minutes on a normal connection. It is safe to re-run.

### Step 3 — confirm it worked

`setup.bat` prints a confirmation at each stage. You should see all of these:

```
[ok] Found Python 3.11
[ok] NVIDIA GPU detected - installing CUDA build of PyTorch
[ok] torch 2.6.0+cu124 CUDA NVIDIA GeForce RTX 4070 Laptop GPU
[ok] Model ready at C:\Users\...\.cache\huggingface\hub\models--ARTPARK-IISc--SraVaani-1.0\...
```

and then the self-test, which must end with:

```
44/44 checks passed
```

The two lines that matter most:

- `[PASS] running on GPU    cuda / fp16` — it is using the GPU, not the CPU.
- `[PASS] silence produces no text` — the noise gate is working.

If the count is not 44/44, the failing check is named on the last line.
See **Troubleshooting** below.

### Step 4 — launch

Double-click **`run.bat`**.

Wait for the badge in the top right to turn to a black **READY** chip showing
`CUDA / fp16`. That takes a few seconds while the model loads onto the GPU.

### Step 5 — confirm dictation end to end

1. Open Notepad and click in it so the cursor is blinking.
2. Hold **Right Shift**, say *"this is a test of the dictation system"*, release.
3. A white pill appears at the bottom of the screen while you speak, showing a
   live waveform, then `TRANSCRIBING`.
4. The text appears in Notepad at your cursor.

If the text appears in the app window but not in Notepad, see the focus note in
Troubleshooting.

---

## Using it

| Shortcut | Action |
|---|---|
| **Hold Right Shift** | Talk. Release to transcribe and paste. |
| **F9** | Toggle dictation on/off (for long passages). |
| **F11** | Paste the last transcript again. |
| **Esc** | Cancel the recording in progress. |

All four are re-bindable in **Settings → Shortcuts**.

### Tabs

- **Dictate** — the latest transcript, plus a history of the session. Double-click
  any history row to copy it.
- **Notes** — a notepad. Tick *Dictate into this note* and everything you say is
  appended there (optionally timestamped) instead of being pasted into another
  app. Save as `.txt` or `.md`.
- **Settings** — microphone, noise filtering, output behaviour, shortcuts, and
  custom vocabulary.

### Language

The **Language** dropdown sits at the top of the Dictate tab.

**Auto-detect is the default.** The model decides the language for each
utterance on its own. Selecting a language from the dropdown **overrides** that
and locks decoding to the chosen script.

This is not a cosmetic setting. SraVaani has no language input, so it cannot be
*told* which language to expect. Instead, selecting a language masks every
token outside that language's script out of the decoder's vocabulary, so the
model physically cannot emit another script.

**Auto-detect is sticky across the session.** Long utterances (4 s or more) set
the session language; short ones inherit it. Changing the session language
requires two long utterances in a row agreeing, so one odd result cannot flip
you into the wrong language mid-dictation.

Use it when auto-detect drifts. Speaking English with an Indian accent, for
example, sometimes lands in Devanagari ("hello" becoming "हेलो"). Selecting
**English** stops that completely.

Each transcript is tagged with the detected language. If you selected one
language and a different script comes out, the tag turns black with a `!` to
flag the mismatch.

### Custom vocabulary

Settings → Vocabulary. One term per line. This is what fixes domain words —
the model writes "sravani" or "art park", and the vocabulary layer rewrites
them to `SraVaani` and `ARTPARK`.

---

## How it works

```
microphone (always open, 16 kHz mono)
   |
   +-- 400 ms pre-roll ring buffer      so an early first word is not clipped
   +-- ambient noise profile            learned continuously while idle
   |
[Right Shift held] --> capture
   |
   +-- high-pass 80 Hz                  removes mains hum and desk thumps
   +-- SNR-adaptive spectral gating     off above 22 dB, gentle to 6 dB, firm below
   +-- WebRTC VAD trim + speech gate    silence never reaches the model
   +-- auto gain                        rescues quiet microphones
   |
SraVaani-1.0 (TorchScript, fp16, CUDA)  greedy TDT decode
   |
   +-- script mask (if a language set)  out-of-script tokens made unreachable
   +-- pause-based punctuation          from the model's own word timestamps
   +-- compound merge, contractions     "to morrow" -> "tomorrow"
   +-- filler removal, casing           Latin-script text only
   +-- custom vocabulary                domain spellings
   |
clipboard -> restore previous window -> Ctrl+V at the caret
```

Four design decisions worth knowing:

**The noise filter is tuned by measurement, not by intuition.** Every policy was
benchmarked on six sentences against six disturbances (white hiss, mains hum,
fan rumble, babble, impulses, keyboard clatter) at 20/15/10/5/0 dB SNR —
180 measurements per policy. Results:

| Policy | Mean WER | Mean WER (SNR <= 10 dB) |
|---|---|---|
| no filtering | 0.0794 | 0.1241 |
| fixed-strength gating | 0.0722 | 0.1162 |
| **SNR-adaptive gating (shipped)** | **0.0710** | **0.1143** |
| decision-directed Wiener filter | 0.0966 | — |

The shipped policy skips filtering entirely above 22 dB SNR, applies gentle
gating (0.5) down to 6 dB, and stronger gating (0.8) below that.

Three things were built, measured, and **deleted for making it worse**: a
decision-directed Wiener filter (0.0966 — classic speech enhancement optimises
perceptual quality, not recogniser accuracy), an impulse suppressor (keyboard
clatter went 0.030 to 0.088), and mains-hum notch filtering (no measurable gain
over the high-pass already in place).

Babble — other people talking — is not improvable by any spectral method,
because it is speech competing with speech. At 0 dB babble every policy scores
near 1.0 WER. If the room is that loud, move the microphone closer; no filter
will save it. Reproduce any of this with `tests/bench_policy.py`.

**Punctuation comes from timestamps, not an LLM.** SraVaani emits no punctuation
or capitalisation. Rather than adding a cloud LLM cleanup step (an API key and a
network round-trip — the things that fail during a live demo), sentence breaks
are inferred from pauses in the model's own word-level timestamps. Devanagari,
Bengali, Gurmukhi, Odia and Gujarati get the danda (।); every other script gets
a full stop.

**Language selection is enforced in the decoder.** Choosing a language builds a
mask over the 5000-token vocabulary and applies it to the logits at every
decoding step, so out-of-script tokens are unreachable.

**Short English utterances used to come out in Hindi, and likelihood scoring
could not fix it.** Dictating "hello hello hello" produced `हेलो हेलो हेलो`.
That is not a malfunction: हेलो is the standard Hindi spelling of the loanword,
and the model prefers Devanagari for it by a wide margin (-0.10 vs -0.52
log-probability per decoding step). Scoring every script and taking the best
still picks Hindi, because Hindi genuinely is the better explanation of one
second of audio. What separates the two is context, not acoustics -- hence the
sticky session language. Measured on Indian-accented English (Microsoft Heera
and Ravi voices) across short, very short, noisy and reverberant variants:
**87.5% correct without session memory, 100% with it.** Reproduce with
`tests/lid_session.py`.

**The NLP cleanup measurably fixes real errors.** On a five-sentence English
test set the model scored 0.125 WER — and every single error was a split
compound: "tomorrow" as "to morrow", "everyone" as "every one", "today" as
"to day". The model heard the audio correctly and mis-segmented the words.
Merging split compounds against a known-word list, plus contraction and
confusion fixes, took that set to **0.000 WER**. Repeat-word collapse and
whitespace normalisation run for every script; filler removal, casing and
compound merging apply to Latin text only, since they would corrupt Indic
output.

---

## Troubleshooting

**Self-test says `model failed to load` / 401 / gated**
You have not accepted the terms, or `.env` has a bad token. Redo Step 1.

**`running on GPU` fails, shows `cpu / fp32`**
`nvidia-smi` was not found during setup, so the CPU build of PyTorch was
installed. Delete the `.venv` folder and re-run `setup.bat`.

**Text is copied but not pasted into the other app**
Windows refused the focus change. The transcript is still on your clipboard —
press Ctrl+V. This usually happens when pasting into an app running as
administrator while this one is not.

**No speech detected, every time**
Wrong microphone. Settings → Input → Microphone, and watch the INPUT LEVEL
meter in the sidebar while you speak — the bars must move.

**Right Shift does nothing**
Another app has claimed it. Settings → Shortcuts → Hold to talk, and pick F8
instead. (Right Ctrl is deliberately not the default: on many new laptops it is
the Copilot key and never reaches this app.)

**English comes out in Hindi script**
Dictate one full English sentence first — that sets the session language and
short utterances then follow it. If it still happens, set the Language dropdown
to **English**, which locks decoding to Latin script outright.

**Indic text shows as boxes**
The Nirmala UI font is missing. Windows Settings → Time & Language → Language,
and add the language pack for the script you need.

---

## Testing

```
.venv\Scripts\python.exe selftest.py        # 18 checks, ~40 s, run by setup.bat
.venv\Scripts\python.exe tests\test_live.py # 23 checks, ~3 min
```

`selftest.py` covers the audio gate, the NLP cleanup in Hindi, Kannada, Telugu
and Tamil, the UI, and the Notes workspace. `test_live.py` covers noise
robustness (white hiss, mains hum, fan rumble at 20/10/5 dB), audio edge cases
(silence, clipping, DC offset, empty buffers), and real cross-process pasting
into another window's text field in English, Kannada, Hindi and Telugu.

Do not type or click while `test_live.py` runs — it takes over the keyboard
focus to test pasting, and stray keystrokes land in the test's target window.

---

## Layout

```
main.py                  launch
setup.bat / run.bat      install / start
selftest.py              fast checks
requirements.txt
sravaani_flow/
    app.py               UI and controller
    engine.py            model loading, worker thread, junk gate
    audio.py             capture, noise filtering, VAD
    cleanup.py           punctuation, fillers, compounds, vocabulary
    decoding.py          script-masked greedy decoding
    translit.py          Devanagari to Latin fallback
    inject.py            focus capture, restore, paste at caret
    hotkeys.py           global shortcuts
    overlay.py           floating recording pill
    languages.py         supported languages and script detection
    theme.py             monochrome styling
    config.py            settings persistence
    fetch.py             model download
tests/
    test_live.py         noise, edge cases, cursor targeting
    bench_noise.py       filtered vs unfiltered WER
    bench_policy.py      denoising policy comparison
    lid_session.py       language detection with session memory
    lid_stress.py        language detection under degradation
    target_window.py     helper window for the paste test
```
