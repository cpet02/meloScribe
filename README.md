# meloScribe

Transcribes the sung melody out of a song: separates the stems, estimates the
melody line with an ensemble of pitch estimators, and attaches time-synced
lyrics fetched automatically from [LRClib](https://lrclib.net/).

Output is a note list with a **calibrated confidence per note** — so you can
see which notes to trust — as a table, CSV, JSON, lead sheet, LRC or MIDI.

---

## Setup

### 1. Install PyTorch first

This must come before anything else, or pip resolves a CPU-only build and the
GPU goes unused.

**With an NVIDIA GPU (CUDA 12.4):**

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
```

The `--extra-index-url` is not optional. `--index-url` alone *replaces* PyPI,
and the PyTorch index does not carry `typing_extensions`; pip then falls back
to building it from an sdist whose own build dependency (`flit_core`) is also
absent, and the install dies with a confusing `flit_core` error after a long
download. Adding PyPI as a secondary index fixes it.

**CPU only:**

```bash
pip install torch==2.5.1 torchaudio==2.5.1
```

Verify it took:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You want `2.5.1+cu124 True`. A `+cpu` suffix means the wrong build installed —
re-run the command above with `--force-reinstall`.

### 2. Install meloScribe and its dependencies

```bash
pip install -e .
```

This also puts the `meloscribe` command on your PATH. (`pip install -r
requirements.txt` works too, but skips the command.)

### 3. Optional extras (recommended)

Each is genuinely optional — a missing one disables its feature with a warning
rather than breaking the run — but each measurably improves the result:

```bash
pip install -e ".[extras]"
```

- **torchcrepe** — the strongest single pitch voter, and the only one whose
  confidence is calibrated out of the box.
- **faster-whisper** — generates lyrics for tracks LRClib has never seen.

Model weights (demucs, basic-pitch, CREPE, the forced aligner) download on
first use and are cached afterwards.

### 4. Check the install

```bash
pip install -e ".[dev]"
pytest tests/test_pipeline_units.py tests/test_eval_harness.py tests/test_api.py -q
```

61 tests should pass. To confirm the GPU and the optional extras are all
visible to the app:

```bash
python -c "from meloscribe.api.app import app; from fastapi.testclient import TestClient; print(TestClient(app).get('/api/health').json())"
```

---

## Usage

### Web UI

```bash
uvicorn meloscribe.api.app:app --port 8000
```

Open <http://localhost:8000>. Drop in a file, confirm the track name, run.
Progress is reported per stage and long jobs can be cancelled.

### Command line

```bash
meloscribe song.mp3 --track "Karma Police" --artist Radiohead
```

A track name is required, because lyric lookup without one silently degrades
into guesswork — and the wrong song's lyrics aligned confidently onto your
melody is worse than no lyrics. To skip lyrics entirely:

```bash
python -m meloscribe.cli song.mp3 --no-lyrics --format leadsheet
```

Other common invocations:

```bash
python -m meloscribe.cli song.mp3 --preset max --transpose 9 --midi sax.mid
python -m meloscribe.cli vocals.wav --vocals-only --format csv -o notes.csv
```

| Flag | Purpose |
|---|---|
| `--preset fast\|balanced\|max` | Separation quality vs. time |
| `--transpose 9` | Alto sax in E♭ (2 for B♭ instruments) |
| `--confidence 0.6` | Drop notes below a confidence |
| `--lyrics-mode` | `off` / `lookup` / `align` / `transcribe` |
| `--vocals-only` | Input is already an isolated vocal |
| `--show-voters` | Show the per-voter breakdown per note (the web UI has a checkbox for this under the results table) |

---

## How it works

### Separation

Demucs via its **Python API** rather than the CLI, which exposes `shifts`
(test-time augmentation), `overlap` and `segment` (VRAM control), and gives
real progress callbacks. Default model is `htdemucs_ft`. Results are cached by
**content hash plus parameters**, so changing a setting is a cache miss by
construction and re-running with identical settings costs nothing.

### Pitch: several estimators, fused

Four voters report per-frame salience over a shared 10ms / semitone grid:

| Voter | Contributes |
|---|---|
| **CREPE** | calibrated per-frame f0 confidence (optional, strongest) |
| **basic-pitch** | note + onset posteriorgram |
| **PYIN** | independent f0 and voicing probability |
| **Harmonic template** | how well a synthetic reed spectrum on each candidate explains the observed CQT |

The active set is resolved at runtime by what is installed, because the best
combination is not simply "all of them":

- **CREPE available** -> `crepe + basic_pitch`
- **CREPE missing** -> `basic_pitch + pyin + harmonic_template`

PYIN measurably *hurts* once CREPE is present (0.973 -> 0.964 OA) while
accounting for 29 of 35 seconds of runtime, so it is dropped from that path.
CREPE alone has the best raw note F1 but collapses to 0.727 on re-articulated
notes, because nothing tells it that three repeated notes are not one long
one - basic-pitch's onset matrix is what fixes that.

The harmonic template voter is the "play a saxophone alongside it and listen"
idea, done in the frequency domain. Doing it literally — synthesising audio and
measuring Plomp–Levelt roughness — is slow and, more importantly, nearly
octave-blind: an octave is maximally consonant, so roughness cannot tell C4
from C5, which is the error that most needs catching. Matching a harmonic comb
against the CQT does resolve it, because a wrong-octave template leaves half its
predicted partials sitting on empty spectrum. (The literal roughness version
ships as the `roughness` voter — it was worth measuring rather than assuming,
and it is off by default because it measured weak.)

Voters are then **fused and Viterbi-decoded** rather than thresholded frame by
frame. The transition model encodes what melodies actually do — hold notes,
move by small intervals — so strong evidence either side of a corrupted frame
carries the note through it. Confidence comes from forward–backward: the
posterior probability of the decoded state given the whole signal, which is a
real probability rather than the note amplitude the old pipeline reported.

Key estimation biases the decoder's log-probabilities **but never filters**.
The previous version deleted out-of-key notes outright, which erased every
accidental, blue note and chromatic passing tone in the song.

### Lyrics

LRClib has a documented JSON API, so this is a client, not a scraper. Duration
is matched within 3 seconds, because titles collide constantly across live
cuts, remasters and covers, and a mismatched hit misaligns everything.

Three tiers, best available first:

1. **Synced LRC, refined** — line timestamps are hand-made and often 200–500ms
   adrift, so they are snapped to detected vocal onsets.
2. **Forced alignment** — with only plain lyrics, a CTC model aligns the known
   text to the vocal stem for **word-level** timings. Better than any LRC file:
   line timing says which phrase a note is in, word timing puts a syllable on a
   note. Melisma is handled — several notes can share one word.
3. **Whisper** — with no lyrics at all, transcribe them from the audio.

---

## Accuracy

`meloscribe.eval` is the scoring harness. Nothing gets tuned without a number.

```bash
python -m meloscribe.eval.runner --systems basic_pitch,ensemble
python -m meloscribe.eval.ablate            # per-voter contribution
python -m meloscribe.eval.runner --systems oracle    # harness self-test
```

Measured on the built-in synthetic benchmark (10 cases covering octave traps,
vibrato, breathiness, separation bleed and re-articulated notes), on an
RTX 3060 with CREPE installed:

| System | Overall Acc | Raw Pitch Acc | Voicing FA | Octave err | **Note F1** |
|---|---|---|---|---|---|
| basic-pitch (old pipeline) | 0.877 | 0.955 | 0.368 | 0.001 | 0.471 |
| **ensemble** (`crepe + basic_pitch`) | **0.973** | **0.998** | **0.107** | **0.000** | **0.919** |

Note F1 roughly doubled and false voicing dropped by two thirds. The old
pipeline's weakness was never pitch — it found the right frequency 95.5% of the
time — it was turning frames into notes.

Real timings on the 3060: a 5:53 track separates with `htdemucs_ft` in ~72s,
and the full pipeline (cached stems, lyric lookup) completes in ~42s.

Honest caveats:

- The synthetic set is generated, so it is easy on pitch and heavy on silence.
  It is a regression detector and a failure-mode probe, **not** a substitute for
  real annotated audio. Drop `audio.wav` + `audio.csv` (`time,frequency`) pairs
  into a folder and run `--dataset <folder>` to score on real ground truth
  (vocadito and MedleyDB both use this layout).
- HPSS denoising is **off** by default because it measured neutral-to-harmful
  here — but this set has no percussive bleed, which is the only thing HPSS
  removes. Re-test on real stems before treating that default as settled.
- The harmonic-template voter carried the CREPE-free configuration — it is what
  eliminated PYIN's octave errors — but adds nothing measurable once CREPE is
  present, and is default only on the no-CREPE path. Being straight about it:
  the saxophone-template idea works, and a better-trained model supersedes it.
- Forced alignment quality has **not** been validated against real sung vocals;
  only the mechanism was exercised, on an instrumental. The confidence floor
  that rejects mismatched alignments is uncalibrated for the same reason.

To compare a change against a previous run:

```bash
python -m meloscribe.eval.runner --systems ensemble --output new.json
python -m meloscribe.eval.runner --systems ensemble --baseline new.json
```

---

## Layout

```
meloscribe/
  audio.py         loading and conditioning
  cache.py         content-addressed stage cache
  stems.py         Demucs separation
  key.py           Krumhansl-Schmuckler key estimation
  pitch/
    grid.py        the shared time/pitch grid
    voters.py      the estimators
    fusion.py      log-domain fusion + Viterbi + forward-backward
    engine.py      orchestration and note segmentation
  lyrics/
    lrclib.py      LRClib API client
    align.py       LRC parsing, onset snapping, forced alignment, Whisper
    service.py     tier selection and the track-name gate
  pipeline.py      stage orchestration with weighted progress
  output.py        table / csv / json / leadsheet / lrc / midi
  cli.py           command line
  api/             FastAPI backend + background jobs
  web/             single-page UI
  eval/            scoring harness, synthetic benchmark, ablation
```

`pipeline/` (lower-case, the original package), `main.py` and `app.py` are the
previous implementation, kept for reference. Their 4 failing stemmer tests are
pre-existing: they shell out to a `demucs` executable that is only on `PATH`
when the venv is activated — the fragility that motivated moving to the
Python API.
