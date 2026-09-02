# meloScribe — handoff

Paste this into a new chat to continue work.

---

## What this is

`C:\Users\chris\Documents\meloScribe` — transcribes the sung/lead melody out of
a song: separates stems, estimates the melody with an ensemble of pitch
estimators, and attaches time-synced lyrics fetched from LRClib. Outputs a note
list with a calibrated confidence per note (table / CSV / JSON / lead sheet /
LRC / MIDI), via a CLI and a FastAPI web UI.

Python 3.10, venv at `./venv`, `main` is current and pushed. Hardware: RTX 3060
12GB, `torch 2.5.1+cu124`, CUDA working. Everything below was measured on this
machine, not assumed.

The package is `meloscribe/`. The lower-case `pipeline/`, `main.py` and `app.py`
are the **superseded** original implementation, kept only for reference — their
4 failing tests in `tests/test_stemmer.py` are pre-existing and expected (they
shell out to a `demucs` executable that is only on PATH when the venv is
activated; the new code uses the Python API instead).

## Layout

```
meloscribe/
  audio.py         loading + conditioning (16kHz analysis rate)
  cache.py         content-addressed stage cache (hash of content + params)
  stems.py         Demucs via Python API, htdemucs_ft, GPU, OOM retry
  key.py           Krumhansl-Schmuckler key estimation
  pitch/
    grid.py        shared 10ms x semitone grid (MIDI 36-96) all voters report onto
    voters.py      CREPE, basic-pitch, PYIN, harmonic template, roughness
    fusion.py      log-domain fusion -> Viterbi -> forward-backward confidence
    engine.py      orchestration + note segmentation
  lyrics/
    lrclib.py      LRClib API client (not a scraper), duration matching
    align.py       LRC parsing, onset snapping, CTC forced alignment, Whisper
    service.py     tier selection + the track-name gate
  pipeline.py      stage orchestration with weighted progress
  output.py        renderers
  cli.py           command line
  api/             FastAPI + threaded job store
  web/index.html   single-page UI
  rhythm.py        beat grid, per-note grid deviation, plausibility score
  eval/            scoring harness, synthetic benchmark, voter ablation
```

## How to run and measure

```bash
venv/Scripts/python -m meloscribe.eval.runner --systems basic_pitch,ensemble
venv/Scripts/python -m meloscribe.eval.ablate          # every voter subset
venv/Scripts/python -m meloscribe.eval.runner --systems oracle   # harness self-test, must be 1.000
venv/Scripts/python -m meloscribe.eval.rhythm_eval --verbose     # rhythm: does the score separate good from bad?
venv/Scripts/python -m pytest tests/test_rhythm.py tests/test_pipeline_units.py tests/test_api.py tests/test_eval_harness.py -q
venv/Scripts/python -m uvicorn meloscribe.api.app:app --port 8000
venv/Scripts/meloscribe song.mp3 --track "Name" --artist "Artist"
```

**Current numbers to beat** (10-case synthetic benchmark, RTX 3060, CREPE on):

| System | OA | RPA | VoicingFA | Octave | Note F1 |
|---|---|---|---|---|---|
| basic-pitch (old pipeline) | 0.877 | 0.955 | 0.368 | 0.001 | 0.471 |
| ensemble (`crepe + basic_pitch`) | **0.973** | **0.998** | **0.107** | **0.000** | **0.919** |

Rhythmic plausibility (7 metrical cases, chance-normalised so 0 = random):

| correct | realistic error | onsets smeared | notes shredded | notes merged | scrambled |
|---|---|---|---|---|---|
| **0.917** | 0.668 | 0.335 | 0.162 | 0.670 | 0.128 |

Real timings: 5:53 track separates in ~72s; full pipeline with cached stems ~42s;
beat tracking adds ~4s and is cached.

## Decisions that were expensive to learn — do not silently undo

1. **Measure first.** The harness came before the tuning, and immediately showed
   the old pipeline's problem was never pitch (RPA 0.955) but frame-to-note
   segmentation (F1 0.471). Any accuracy change must be reported as a
   before/after from `eval.runner` or `eval.ablate`, never asserted.
2. **No voter may veto** (`FusionSettings.voter_floor`). Without a floor on each
   voter's salience before the log, a single voter's near-zero score is a ~-18
   log penalty nothing can outvote, and the ensemble scored *below its own best
   member* (0.848 vs 0.920). There is a regression test for this.
3. **The voter set is resolved at runtime**, not fixed
   (`voters.resolve_voter_names`): `crepe + basic_pitch` when CREPE is
   installed, `basic_pitch + pyin + harmonic_template` otherwise. PYIN actively
   hurts once CREPE is present (0.973 -> 0.964) and costs 29 of 35 seconds.
   CREPE alone has the best F1 overall but collapses to 0.727 on re-articulated
   notes — basic-pitch's onset matrix is what fixes that.
4. **Splitting a repeated note needs an amplitude attack, not just an onset
   activation** (`EngineSettings.attack_threshold`). basic-pitch's onset output
   peaks at the vibrato rate, so threshold-only splitting shredded held notes
   into ~200ms fragments at exactly 5Hz. Gating on a rise in the RMS envelope
   took note F1 from 0.318 to 0.899.
5. **Key biases, never filters** (`fusion.key_prior_vector`). The old code
   deleted out-of-key notes, erasing every accidental and blue note. Same
   principle applies to any new musical prior — including the rhythm work below.
6. **Library conversions over hand-rolled ones.** CREPE scored RPA 0.095 because
   its bin-0 frequency was hard-coded as 32.70Hz when it is really ~31.70Hz — a
   54-cent bias, just past the 50-cent scoring tolerance. It now reads
   `torchcrepe.convert.bins_to_frequency`. It looked exactly like a broken model.
7. **A benchmark that cannot see a failure will argue against fixing it.** Every
   synthetic melody originally had an 80ms rest after each note, so nothing
   tested re-articulation, and the ablation "proved" basic-pitch was harmful.
   Adding legato cases reversed the conclusion. Before trusting an ablation, ask
   what the benchmark cannot express.
8. **Reject weak forced alignments** (`MIN_ALIGNMENT_CONFIDENCE`). Forced
   alignment always returns *something*: on an instrumental with invented
   lyrics it produced plausible timings at confidence ~0.05.

## Known gaps

- **Forced alignment is unvalidated on real sung vocals.** Only the mechanism
  was exercised, on an instrumental, so `MIN_ALIGNMENT_CONFIDENCE = 0.15` is
  uncalibrated. If it fires on a song that obviously matches its lyrics, it is
  too aggressive.
- **No real annotated audio has been scored.** The synthetic set is a
  regression detector and failure-mode probe, not a substitute. Drop
  `audio.wav` + `audio.csv` (`time,frequency`) pairs in a folder and run
  `--dataset <folder>`; vocadito and MedleyDB use that layout.
- **HPSS denoising is off by default** (measured neutral-to-harmful) — but the
  synthetic set has no percussive bleed, which is the only thing HPSS removes.
  Untested on real stems.
- **Backing-vocal bleed** (harmonies leaking into the vocal stem) is not
  modelled in the benchmark at all and is a likely real-world failure.
- Beat/tempo tracking exists only in the **legacy** `pipeline/beat_tracker.py`
  and was not carried into `meloscribe/`.
- The web UI does not expose `--device`.

---

## Rhythm work: done, and what it found

`meloscribe/rhythm.py` tracks a beat grid from the **full mix** (not the vocal
stem - separation removes the percussive onsets beat tracking needs), scores
every note's deviation from it, and produces a track-level plausibility score.
**It is off by default** - `--rhythm`, `assess_rhythm=True`, or the UI
checkbox. The default pipeline is byte-identical to before it existed: same
notes, same columns, same runtime. That is deliberate, and the reason is the
validation below: the diagnostic rests on one real track, which is not enough
evidence to spend four seconds and two output columns on every run. It changes
**no notes** even when it is on.

Three things carry the design, and undoing any of them breaks it:

1. **The score is normalised against chance**, per note count, by simulating
   the whole estimator on random input. This is not a refinement - it is the
   metric. Random onsets score ~0.62 on the raw measure, so an un-normalised
   "notes are near the grid" is just as true of noise. Measured floor over 8
   seeds: 0.086 (sd 0.106 at ~16 notes, so a short track's score is noisy).
2. **Two things are fitted per track** - the grid's phase offset (a singer sits
   behind the beat) and whether the beat divides in two or three - and both are
   priced into the chance level. They move the *grid*, never a note.
3. **Binary and ternary divisions are alternatives, not a union.** The union
   was tried first: six positions per beat puts chance at 0.85 and the metric
   stops discriminating. Choosing one per track improved every row of the
   benchmark at once.

Benchmark additions: `synth.METRICAL_CASES` - melodies written in beats at a
stated BPM over a percussive pulse, including swing, rubato, triplets and a
half-time tempo trap. `eval/rhythm_eval.py` scores each against deliberate
corruptions. A wrong metrical level costs real accuracy but does not destroy
the score: correct notes score 0.92 on the tracked grid, 0.68 at half tempo,
0.63 at double.

**A reproducibility bug was fixed on the way**: `synth.py` seeded cases with
Python's `hash()`, which is randomised per process. Cached WAVs stayed fixed
while regenerated annotations drifted, so the rubato case's score moved by 0.18
with no code change. Now `zlib.crc32`. Check for this pattern before trusting
any benchmark number that will not reproduce.

### What validating it against real audio showed

There is exactly one real track in the repo (`samples/sample.mp3`); the 24
files in `data/uploads/` are identical 53KB API-test fixtures. So the flag
could not be calibrated against a corpus, only stress-tested on one track.
Scored in 40-note windows against the same grid:

| | whole track | windows (mean +/- sd) | range | flagged at 0.35 |
|---|---|---|---|---|
| our note onsets | 0.087 | 0.068 +/- 0.083 | 0.00-0.20 | 100% |
| librosa onsets, same stem | 0.422 | 0.365 +/- 0.138 | 0.10-0.64 | 48% |

Two conclusions, pulling in opposite directions:

- **The signal is real and consistent.** Our onsets score worse than a plain
  onset detector in *every* window, not merely on average. That is not a
  sampling artefact.
- **The synthetic-derived threshold was wrong.** At 0.35 the flag fired on
  half the windows of a reference that is not even our transcription. A flag
  that is a coin toss on decent input is worse than none, so `FLAG_THRESHOLD`
  is now 0.10 - anchored to the worst window that baseline produced on real
  music, not to what a sequencer would manage.

Compare *onset_score*, not `plausibility`, when using an onset detector as a
reference: it has no real durations, so its plausibility is unfairly penalised
(0.27 against its true 0.42).

### The finding worth acting on

On the one real track available (`samples/sample.mp3`), the transcription
scores **0.12** and is flagged. That looks like a true positive, not a
miscalibration. On the same beat grid:

| what was scored | onset score |
|---|---|
| notes placed exactly on the grid | 1.000 |
| librosa's onsets on the vocal stem | 0.422 |
| **our transcribed note onsets** | **0.123** |

The grid is sound and the singer is reasonably on the beat; our note onsets are
not. Deviation spread is 0.065 beat (~29ms) against 0.052 for the independent
detector, with no drift across the track. Note the comparison is between
*distributions* - the onset detector fires 451 times against our 162 notes, so
it is not note-for-note.

This is a lead the pitch benchmark could not produce: synthetic note F1 is
0.919 and says nothing is wrong. The next accuracy work is probably onset
placement in `engine._segment` - the attack gate decides *whether* to split,
but nothing refines *where*.

### Known gaps in the rhythm work

- **`FLAG_THRESHOLD = 0.10` is a floor, not a calibration.** It is anchored to
  one baseline on one track. That is enough to stop it crying wolf; it is not
  enough to know what it should be. Widen the evidence before raising it - and
  the way to do that is real annotated audio, per the gap above.
- **Under-segmentation is nearly invisible to it.** Merged notes start on the
  grid and last the sum of two grid values, which is usually a grid value:
  merging every second note scored 0.67 against 0.92 correct. A reference-free
  measure cannot miss what was never emitted.
- **It is not fed back into the decoder.** Step 4 of the original plan (a weak
  log-prior on note boundaries near subdivisions) is not done, deliberately -
  it should not be attempted until the onset-placement finding above is
  understood, since biasing onsets toward the grid would paper over exactly the
  defect this just uncovered.
- **The tempo trap did not spring.** `metrical_halftime` renders a 132 BPM
  melody over a 66 BPM backbeat and librosa still found 132, so the octave
  diagnostics are exercised only against synthetically halved and doubled
  grids, never against a tracker that actually erred.

## Working agreements

- Numbers before claims: run the harness before and after, and report the delta.
- Musical knowledge enters as a prior, never as a filter.
- Prefer a library's own conversions to hand-derived constants.
- Optional dependencies degrade with a warning, never break the run.
- Ask what the benchmark cannot see before trusting what it says.
