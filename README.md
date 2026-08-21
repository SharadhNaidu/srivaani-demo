# SraVaani Flow

offline push-to-talk dictation on the ARTPARK-IISc SraVaani-1.0 speech
recognition model. hold a key, speak, release — the text lands at your cursor in
whatever app you were typing in.

runs entirely on your machine. no accounts, no tokens, no api keys at any point.
after the one-time model download there are no network calls at all.

---

## Install — three steps

**1. clone the repo**

```
git clone https://github.com/SharadhNaidu/srivaani-demo
```

**2. run `setup.bat`**

double-click it, or run it from a terminal inside the folder you just cloned.
it creates a virtual environment, installs pytorch (the cuda build if it finds
an nvidia gpu, otherwise the cpu build), downloads the model, and finishes by
running `verify.py`.

this takes 5–15 minutes on a normal connection. it is safe to re-run — the
model download resumes where it left off.

**3. run `run.bat`**

that's it. wait for the badge in the top right to turn into a black **READY**
chip — it shows `CUDA / fp16` on a gpu machine or `cpu / fp32` on a cpu one.
on cpu the model takes about 10 seconds to load.

> do **not** run `pip install -r requirements.txt` on its own. pytorch is
> deliberately not listed there — `setup.bat` installs it first, from nvidia's
> cuda wheel index when a gpu is present. installing it from pypi instead gives
> you the cpu build, and the gpu is silently lost.

### what you should see

`setup.bat` ends by running `verify.py`, which prints one line per check:

```
  [ok]   python version                         3.11
  [ok]   all packages import
  [ok]   torch build                            2.6.0+cu124
  [ok]   model loaded                           SharadhNaiduTrains/sravaani-flow-model
  [ok]   transcribed the sample
  [warn] no microphone detected                 plug one in before dictating; setup is still fine
```

- `[ok]` — that check passed.
- `[warn]` — worth knowing, but setup still completes. **a missing microphone is
  only a warning**, so a machine with no mic attached still finishes setup
  cleanly. so is a missing gpu.
- `[FAIL]` — setup stops and names the problem on the last line. see
  **Troubleshooting**.

### your first dictation

1. open notepad and click in it so the cursor is blinking.
2. hold **Right Shift**, say *"this is a test of the dictation system"*, release.
3. a white pill appears at the bottom of the screen while you speak, showing a
   live waveform, then `TRANSCRIBING`.
4. the text appears in notepad at your cursor.

if the text appears in the app window but not in notepad, see the focus note in
**Troubleshooting**.

---

## Requirements

| Requirement | How to check | If it's missing |
|---|---|---|
| Windows 10 or 11 | — | — |
| Python 3.10, 3.11 or 3.12 | run `py -0p` | install from python.org. **not 3.13+** — pytorch has no wheels for it yet |
| ~4 GB free disk | — | the model is about 870 MB, pytorch cuda is about 2.5 GB |
| NVIDIA driver 527.41 or newer | run `nvidia-smi` | **only needed if you want gpu acceleration.** the pinned cuda 12.4 build needs 527.41+ on windows |

every dependency is pinned to an exact version in `requirements.txt`, and
pytorch is pinned to `2.6.0+cu124` on gpu machines, so a laptop set up next
month gets the same versions this was built and tested on.

---

## A GPU is optional

this runs fine without one. the cpu path is not a fallback that limps — it was
measured, and it is fast enough for dictation.

| Machine | Real-time factor | Notes |
|---|---|---|
| RTX 4070 laptop GPU | 0.10 – 0.20 | uses about 0.94 GB VRAM |
| CPU, 24 threads | 0.10 – 0.13 | roughly 10x faster than real time |

real-time factor is processing time divided by audio length, so 0.10 means one
second of speech is transcribed in about a tenth of a second. on cpu the model
takes about 10 seconds to load at startup; after that, dictation feels the same.

`setup.bat` picks the right pytorch build for you — it looks for `nvidia-smi`
and installs the cuda build if it finds one, the cpu build (about 250 MB) if it
does not. you can also force the cpu path at any time with
Settings → Compute → `cpu`.

---

## The model

the app downloads from **`SharadhNaiduTrains/sravaani-flow-model`**, a public,
mit-licensed, byte-identical mirror. no hugging face account, sign-in or token
is involved — the download is anonymous.

all credit for the model belongs to **ARTPARK-IISc**. the canonical source is
<https://huggingface.co/ARTPARK-IISc/SraVaani-1.0>; the mirror exists only so
that setup needs zero manual steps.

---

## Using it

| Shortcut | Action |
|---|---|
| **Hold Right Shift** | talk. release to transcribe and paste. |
| **F9** | toggle dictation on/off (for long passages). |
| **F11** | paste the last transcript again. |
| **Esc** | cancel the recording in progress. |

all four are re-bindable in **Settings → Shortcuts**.

### Tabs

- **Dictate** — the latest transcript, plus a history of the session.
  double-click any history row to copy it.
- **Notes** — a notepad. tick *Dictate into this note* and everything you say is
  appended there (optionally timestamped) instead of being pasted into another
  app. save as `.txt` or `.md`.
- **Settings** — microphone, noise filtering, output behaviour, shortcuts, and
  custom vocabulary.

### Language

the **Language** dropdown sits at the top of the Dictate tab.

**auto-detect is the default.** the model decides the language for each
utterance on its own. selecting a language from the dropdown **overrides** that
and locks decoding to the chosen script.

this is not a cosmetic setting. SraVaani has no language input, so it cannot be
*told* which language to expect. instead, selecting a language masks every token
outside that language's script out of the decoder's vocabulary, so the model
physically cannot emit another script.

**auto-detect is sticky across the session.** long utterances (4 s or more) set
the session language; short ones inherit it. changing the session language
requires two long utterances in a row agreeing, so one odd result cannot flip
you into the wrong language mid-dictation.

use the dropdown when auto-detect drifts. speaking english with an indian
accent, for example, sometimes lands in devanagari ("hello" becoming "हेलो").
selecting **English** stops that completely.

each transcript is tagged with the detected language. if you selected one
language and a different script comes out, the tag turns black with a `!` to
flag the mismatch.

### Custom vocabulary

Settings → Vocabulary. one term per line. this is what fixes domain words — the
model writes "sravani" or "art park", and the vocabulary layer rewrites them to
`SraVaani` and `ARTPARK`.

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
SraVaani-1.0                            greedy TDT decode
   |                                    fp16 on CUDA, fp32 on CPU
   +-- script mask (if a language set)  out-of-script tokens made unreachable
   +-- pause-based punctuation          from the model's own word timestamps
   +-- compound merge, contractions     "to morrow" -> "tomorrow"
   +-- filler removal, casing           Latin-script text only
   +-- custom vocabulary                domain spellings
   |
clipboard -> restore previous window -> Ctrl+V at the caret
```

four design decisions worth knowing.

### the noise filter is tuned by measurement, not by intuition

every policy was benchmarked on six sentences against six disturbances (white
hiss, mains hum, fan rumble, babble, impulses, keyboard clatter) at
20/15/10/5/0 dB snr — 180 measurements per policy. results:

| Policy | Mean WER | Mean WER (SNR <= 10 dB) |
|---|---|---|
| no filtering | 0.0794 | 0.1241 |
| fixed-strength gating | 0.0722 | 0.1162 |
| **SNR-adaptive gating (shipped)** | **0.0710** | **0.1143** |
| decision-directed Wiener filter | 0.0966 | — |

the shipped policy skips filtering entirely above 22 dB snr, applies gentle
gating (0.5) down to 6 dB, and stronger gating (0.8) below that.

three things were built, measured, and **deleted for making it worse**: a
decision-directed wiener filter (0.0966 — classic speech enhancement optimises
perceptual quality, not recogniser accuracy), an impulse suppressor (keyboard
clatter went 0.030 to 0.088), and mains-hum notch filtering (no measurable gain
over the high-pass already in place).

babble — other people talking — is not improvable by any spectral method,
because it is speech competing with speech. at 0 dB babble every policy scores
near 1.0 wer. if the room is that loud, move the microphone closer; no filter
will save it. reproduce any of this with `tests/bench_policy.py`.

### punctuation comes from timestamps, not an LLM

SraVaani emits no punctuation or capitalisation. rather than adding a cloud llm
cleanup step (an api key and a network round-trip — the things that fail during
a live demo), sentence breaks are inferred from pauses in the model's own
word-level timestamps. devanagari, bengali, gurmukhi, odia and gujarati get the
danda (।); every other script gets a full stop.

### language selection is enforced in the decoder

choosing a language builds a mask over the 5000-token vocabulary and applies it
to the logits at every decoding step, so out-of-script tokens are unreachable.

### short English utterances used to come out in Hindi, and likelihood scoring could not fix it

dictating "hello hello hello" produced `हेलो हेलो हेलो`. that is not a
malfunction: हेलो is the standard hindi spelling of the loanword, and the model
prefers devanagari for it by a wide margin (-0.10 vs -0.52 log-probability per
decoding step). scoring every script and taking the best still picks hindi,
because hindi genuinely is the better explanation of one second of audio. what
separates the two is context, not acoustics — hence the sticky session language.
measured on indian-accented english (microsoft heera and ravi voices) across
short, very short, noisy and reverberant variants: **87.5% correct without
session memory, 100% with it.** reproduce with `tests/lid_session.py`.

### the NLP cleanup measurably fixes real errors

on a five-sentence english test set the model scored 0.125 wer — and every
single error was a split compound: "tomorrow" as "to morrow", "everyone" as
"every one", "today" as "to day". the model heard the audio correctly and
mis-segmented the words. merging split compounds against a known-word list, plus
contraction and confusion fixes, took that set to **0.000 WER**. repeat-word
collapse and whitespace normalisation run for every script; filler removal,
casing and compound merging apply to latin text only, since they would corrupt
indic output.

---

## Troubleshooting

**`verify.py` says `[warn] no gpu detected`**
that is not an error. the cpu build is installed and runs about 10x faster than
real time. if you *do* have an nvidia gpu and want to use it, check that
`nvidia-smi` runs in a terminal (if not, install or update the nvidia driver),
then delete the `.venv` folder and re-run `setup.bat`.

**CUDA errors on an older machine**
the pinned cuda 12.4 build needs nvidia driver 527.41 or newer on windows.
update the driver, or set Settings → Compute → `cpu` to run without the gpu.

**`ImportError` mentioning win32 / pywin32**
some systems need the pywin32 post-install step. `verify.py` warns about this
and prints the fix. run:
`.venv\Scripts\python.exe .venv\Scripts\pywin32_postinstall.py -install`

**Text is copied but not pasted into the other app**
windows refused the focus change. the transcript is still on your clipboard —
press Ctrl+V. this usually happens when pasting into an app running as
administrator while this one is not.

**No speech detected, every time**
wrong microphone. Settings → Input → Microphone, and watch the INPUT LEVEL meter
in the sidebar while you speak — the bars must move.

**Right Shift does nothing**
another app has claimed it. Settings → Shortcuts → Hold to talk, and pick F8
instead. (Right Ctrl is deliberately not the default: on many new laptops it is
the copilot key and never reaches this app.)

**English comes out in Hindi script**
dictate one full english sentence first — that sets the session language and
short utterances then follow it. if it still happens, set the Language dropdown
to **English**, which locks decoding to latin script outright.

**Indic text shows as boxes**
the nirmala ui font is missing. Windows Settings → Time & Language → Language,
and add the language pack for the script you need.

---

## Testing

```
.venv\Scripts\python.exe verify.py          # install check, run by setup.bat
.venv\Scripts\python.exe selftest.py        # fast checks, ~40 s
.venv\Scripts\python.exe tests\test_live.py # full live checks, ~3 min
```

`selftest.py` covers the audio gate, the nlp cleanup in hindi, kannada, telugu
and tamil, the ui, and the notes workspace. `test_live.py` covers noise
robustness (white hiss, mains hum, fan rumble at 20/10/5 dB), audio edge cases
(silence, clipping, dc offset, empty buffers), and real cross-process pasting
into another window's text field in english, kannada, hindi and telugu.

do not type or click while `test_live.py` runs — it takes over the keyboard
focus to test pasting, and stray keystrokes land in the test's target window.

---

## Layout

```
main.py                  launch
setup.bat / run.bat      install / start
verify.py                install check, run at the end of setup
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
    config.py            settings persistence, model repo
    fetch.py             model download
tests/
    test_live.py         noise, edge cases, cursor targeting
    bench_noise.py       filtered vs unfiltered WER
    bench_policy.py      denoising policy comparison
    lid_session.py       language detection with session memory
    lid_stress.py        language detection under degradation
    target_window.py     helper window for the paste test
```
