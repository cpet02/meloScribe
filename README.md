# meloScribe

A command-line tool that transcribes the melody from an MP3 file into musical notes.

meloScribe isolates the vocal/melody track using AI stem separation, detects pitch with high confidence, maps frequencies to note names, and optionally aligns detected notes to timestamped lyrics. Output is transposed for **alto saxophone (Eb)** by default, so the notes are ready to play without any manual transposition.

---

## How It Works

```
MP3 input
    ↓
[1] Stem Separation   →  isolated vocals.wav  (Demucs)
    ↓
[2] Pitch Detection   →  frequency + confidence per note  (basic-pitch)
    ↓
[3] Note Mapping      →  Hz → note name, filtered by confidence
    ↓
[4] Lyric Alignment   →  match notes to lyric lines via .lrc file (optional)
    ↓
[5] Output            →  table / CSV / JSON / leadsheet
```

---

## Requirements

- Python 3.11
- FFmpeg (required by Demucs for reading MP3 files)

### Installing FFmpeg (Windows)

```bash
winget install ffmpeg
```

Or download from https://ffmpeg.org/download.html — use the **full-shared** build from gyan.dev and add the `bin` folder to your system PATH.

Verify it works:
```bash
ffmpeg -version
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourname/meloScribe.git
cd meloScribe
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

**3. Install dependencies**
```bash
pip install --upgrade pip setuptools
pip install "setuptools<70"
pip install -r requirements.txt
```

> **Note:** The `setuptools<70` pin is required because `basic-pitch` depends on `resampy`, which uses `pkg_resources`. Newer setuptools versions moved this module in a way that breaks the import.

**4. Verify the installation**
```bash
python -c "import basic_pitch; print('basic-pitch ok')"
python -c "import demucs; print('demucs ok')"
```

---

## Usage

### Basic usage (stems the MP3, outputs a table in alto sax pitch)
```bash
python main.py --input samples/sample.mp3
```

### Skip re-stemming if you already have a vocals file
```bash
python main.py --vocals "separated/htdemucs/sample/vocals.wav" --confidence 0.75
```

### Leadsheet output with lyrics, saved to a text file
```bash
python main.py --vocals "separated/htdemucs/sample/vocals.wav" \
               --lyrics samples/sample.lrc \
               --confidence 0.75 \
               --format leadsheet \
               --output results.txt
```

### Concert pitch output (no transposition)
```bash
python main.py --input samples/sample.mp3 --transpose 0
```

---

## All Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | — | Path to MP3 file |
| `--vocals` | — | Path to pre-stemmed vocals.wav (skips Phase 1) |
| `--lyrics` | — | Path to `.lrc` timestamped lyrics file (optional) |
| `--confidence` | `0.85` | Confidence threshold 0–1. Lower = more notes, less certain |
| `--transpose` | `9` | Semitones to transpose. `9` = alto sax Eb, `0` = concert pitch |
| `--format` | `table` | Output format: `table`, `csv`, `json`, or `leadsheet` |
| `--output` | — | File path to save output. Prints to terminal if omitted |

> Either `--input` or `--vocals` must be provided.

---

## Output Formats

**table** — formatted terminal grid, best for quick inspection

**leadsheet** — notes grouped by lyric line with dash spacing that reflects timing gaps between notes. Closest to a hand transcription. Requires `--lyrics` for best results.

**csv / json** — structured data output, useful for further processing

### Example leadsheet output
```
I finally found the time to write you this letter
D5 D5 - A#4 -- A#4 - A#4 A#4 A#4 - A#4 A#4 - A#4 - A4 A4 --- G4

And it's coming back soon
E4 -- E4 - E4
```

Dash spacing guide: ` ` = <0.3s gap, ` - ` = 0.3–0.8s, ` -- ` = 0.8–2s, ` --- ` = >2s

---

## Lyrics Files (.lrc)

meloScribe accepts standard `.lrc` format with millisecond timestamps:
```
[00:00.71] Picture me better
[00:05.22] I finally found the time to write you this letter
```

LRC files for most songs can be found at https://lrclib.net

---

## Confidence Threshold Guide

| Threshold | Behavior |
|---|---|
| `0.90`+ | Very few notes, high accuracy |
| `0.85` | Default — sparse but reliable |
| `0.75` | Good balance for clean vocal tracks |
| `0.70` | More coverage, occasional noise |

Start at `0.75` for stripped-back recordings (acoustic, sparse production).
Use `0.85`+ for dense mixes where stem separation is noisier.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Phase 2 tests require the vocals.wav output from Phase 1. Run a stem separation first:
```bash
python main.py --input samples/sample.mp3
```

Then run tests:
```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
meloScribe/
├── main.py                  # CLI entry point
├── requirements.txt
├── README.md
├── pipeline/
│   ├── stemmer.py           # Phase 1 — Demucs vocal isolation
│   ├── pitch_detector.py    # Phase 2 — basic-pitch pitch detection
│   ├── note_mapper.py       # Phase 3 — Hz → note, filtering, transposition
│   ├── lyric_aligner.py     # Phase 4 — LRC parsing and note alignment
│   └── output.py            # Phase 5 — table/CSV/JSON/leadsheet formatter
├── tests/
│   ├── test_stemmer.py
│   ├── test_pitch_detector.py
│   ├── test_note_mapper.py
│   ├── test_lyric_aligner.py
│   └── test_output.py
└── samples/
    ├── sample.mp3           # Place your test MP3 here
    └── sample.lrc           # Optional matching LRC lyrics file
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `demucs` | AI stem separation (vocals isolation) |
| `basic-pitch` | Spotify's pitch detection model |
| `librosa` | Audio analysis and note conversion |
| `soundfile` | WAV file I/O |
| `torchaudio<2.6` | Audio backend for Demucs |
| `numpy` | Numerical processing |
| `pandas` | Data handling |
| `tabulate` | Terminal table formatting |
| `pytest` | Test runner |