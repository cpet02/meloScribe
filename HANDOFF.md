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
  sections.py      splits the notes into lyric lines / song parts / phrases
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
| ensemble, before onset work | 0.973 | 0.998 | 0.107 | 0.000 | 0.919 |
| ensemble (`crepe + basic_pitch`) | **0.970** | **0.997** | **0.116** | **0.000** | **1.000** |

The benchmark is now 13 cases, not 10 - the three added ones are harder, which
is why OA moves a hair while note F1 goes to 1.000 on every case.

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
- The web UI has a **Simple / Advanced** toggle (remembered in localStorage).
  Advanced controls stay in the DOM when hidden, so simple mode submits the
  identical request with the defaults left alone - there is one pipeline and
  one request shape, not two code paths. If you add an option, add it to the
  advanced markup with a sane default and simple mode inherits it for free.

---

## Onset placement: fixed, and how it was found

Note F1 is **1.000 on all 13 cases**, up from 0.919. The defect was not where
the rhythm score pointed and not what it looked like from the outside.

The chain, because the order is the lesson:

1. The rhythm diagnostic said our real-track onsets were far worse than a plain
   onset detector's on the same stem.
2. The synthetic benchmark disagreed: note F1 at a *25ms* tolerance was
   identical to F1 at 50ms, and onset bias was -2ms. Every synthetic note
   started in 20ms at the right pitch, so onset placement was trivial and the
   benchmark could not express the question.
3. Adding `soft` and `scooped` voices - slow attacks, and a 140-cent scoop up
   to pitch over 90ms, which is what singers actually do - dropped note F1 to
   0.727.
4. But recall stayed at **1.000** and onset MAE at **4-7ms**. Nothing was
   misplaced or missed. Precision was 0.571: the notes were *spurious*. A
   one-second closing note was coming out as five, spaced 0.2s apart - 5Hz,
   the vibrato rate.

So the real defect was over-segmentation, and the cause was that
`_attack_envelope` normalised the amplitude rise by the 99th percentile of
rises across the track. That silently assumes the track contains a hard attack
somewhere to set the scale by. On softly-sung material there is none, the scale
collapses to the size of the vibrato ripple, and the ripple clears the gate.
Reproduced directly: under the old envelope one held soft note segmented into
**ten** fragments.

The fix is to measure the attack as how far the level has climbed out of its
recent trough *as a fraction of the current level* - scale-free, so the same
threshold means the same thing whether the singer punches or breathes.
`attack_threshold` moves 0.12 -> 0.45, swept: every case scores 1.000 across
0.35-0.45, soft material shreds below it (0.80 at 0.12) and genuine
re-articulations start being missed above it (0.92 at 0.55). The old 0.12 was
tuned against the old envelope and does not mean the same thing now.

Independent confirmation on the real track, which shares no code path with the
synthetic set: rhythm onset score **0.087 -> 0.154**, flagged windows 100% ->
67%, and 9 fewer notes.

That fix is real and stands. The remaining gap to the real-audio baseline turned
out to be something else entirely - see the next section, which also corrects
two claims made here earlier.

`tests/test_segmentation.py` guards both edges. Note the tempting test that
does *not* work: checking that a quieter copy of the same audio segments
identically. The old percentile normalisation was already gain-invariant; what
it lacked was invariance to attack *shape*. Parametrising over timbres is what
catches it.

## The rest of the gap: it is voicing coverage, not onset placement

Chased to a conclusion. Two things I previously wrote are wrong, and the record
is corrected here rather than quietly edited above.

**The remaining defect: we transcribe 42% of the singing.** The decoder marks
18% of the track voiced while the vocal stem is loud for 40%. What comes out is
152 short fragments (median 0.25s) covering under half the sung material, and
their start times agree with real vocal articulations **at chance**: 38.8% land
within 80ms of one, against 37.0% +/- 3.8% for random times drawn from the same
loud-vocal regions. Onset *placement inside what we do detect* is fine - 4-7ms
MAE on synthetic, +21ms median on real near-matches. So "onset placement" was
the wrong name for this. It is a recall problem wearing a timing problem's
clothes, and the rhythm score could not tell the difference because a fragment
that starts in the middle of a phrase is off-grid exactly like a mistimed note.

**Correction 1: the 0.42 baseline is not inflated by drum bleed.** I suspected
it was, since drums are on-grid by construction and separation is not surgical.
It is not: excluding vocal-stem onsets that coincide with a hit in *any* other
stem leaves 187 onsets that score **0.682**, higher than the ones that do
coincide (0.249). The cleanest vocal onsets are the most grid-aligned. The
singer is genuinely, strongly on the beat and the gap is entirely ours.

**Correction 2: it is not portamento or rounding flips at legato boundaries.**
That was my hypothesis and the data refuses it. Whether a boundary lands on an
articulation is flat across pitch-step size (1 semitone 41%, 2 semitones 50%,
3+ 45%) and flat between legato and after-rest boundaries (46% vs 41%) - all of
which is just the chance rate. There is no time-base error either: no
consistent lag (best lags per quarter scatter -0.99, -0.04, -0.47, -0.24s) and
the reported duration matches the file exactly.

**It is not reachable with the knobs that exist.** Both were swept in both
directions, against synthetic false-alarm rate and real coverage together:

| `silence_bias` | synth VFA | note F1 | real coverage | real onset score |
|---|---|---|---|---|
| -1.0 | 0.037 | 1.000 | 14% | 0.142 |
| **0.0** | **0.116** | **1.000** | **42%** | **0.156** |
| +0.5 | 0.255 | 0.982 | 68% | 0.124 |
| +1.0 | 0.344 | 0.969 | 91% | 0.040 |

Coverage buys nothing: the frames it wins are not melody. `voicing_switch_penalty`
is worse than useless here - raising it cuts coverage too (42% -> 20% at 8.0),
because it suppresses phrase *entries* as well as exits.

So the next work is upstream, in what the voters report as voicing confidence
on a real separated vocal - reverberant, quiet, artefacted - not in the decoder
that consumes it. Worth measuring per-voter voicing against the loud-vocal mask
before touching anything. And note the whole diagnosis rests on one track, with
an energy gate standing in for an annotation; real annotated audio would settle
it properly and remains the most valuable thing missing from this repo.

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
  log-prior on note boundaries near subdivisions) is still not done, and should
  stay undone: the real track sits at 0.154 against a 0.42 baseline, so there
  is a genuine defect left to find, and biasing onsets toward the grid would
  hide it rather than fix it. Find the remaining 0.27 first.
- **The tempo trap did not spring.** `metrical_halftime` renders a 132 BPM
  melody over a 66 BPM backbeat and librosa still found 132, so the octave
  diagnostics are exercised only against synthetically halved and doubled
  grids, never against a tracker that actually erred.

## Section slider

The web UI steps through the result one section at a time
(`meloscribe/sections.py`, served in `/api/jobs/{id}/notes` as `sections`;
audio for playback from `/api/jobs/{id}/audio/{mix|vocals}`).

- **Invariant: every note is in exactly one section per granularity**, in time
  order (property-tested over random layouts in `tests/test_sections.py`).
  Notes outside every lyric line (ad-libs, hums, bleed in a break) get
  'No lyric' sections of their own rather than disappearing; fragments under
  0.6 s of singing are folded into a near neighbour, never into a section more
  than 10 s long, and never pulling a lyric line's start more than 1.35 s ahead
  of its first word.
- Lines come from the timed lyrics. A note up to 0.15 s before a line's start
  belongs to that line (pickup syllables land a hair early) unless it overlaps
  the previous line more. Plain LRClib lyrics without forced alignment have
  placeholder times, so they are not used to cut: the fallback is phrases split
  at rests >= 0.45 s; phrases > 10 s split at their longest gap (near-ties go
  to the middle, and both halves must keep >= 0.6 s of singing).
- Parts break where consecutive *lyric lines* are >= 2.5 s apart - measured
  between lyric lines, so stray notes in an instrumental break cannot glue the
  break onto the next verse; those notes become a part of their own - and
  around any run of >= 2 lyric lines that repeats elsewhere (how a chorus is
  found). A part > 45 s splits at its longest gap. A part with the same words
  as an earlier one gets `repeat_of`. Without lyrics, parts break at silences.
- A wordless LRC timestamp (`[01:23.45]`, `♪`) now *ends* the line before it
  instead of being dropped, so a line no longer runs through an instrumental
  break. `refine_line_times` keeps those ends. Still unhandled (pre-existing):
  the `[offset:]` tag, and `[00:00.00] Title` lines read as lyrics.
- Seeking needs HTTP Range. Starlette's `FileResponse` only answers Range from
  0.39; `meloscribe/api/media.py` serves ranges itself on older versions.
- All thresholds are judgement calls tested on synthetic layouts only; tune
  them against real songs.

## Working agreements

- Numbers before claims: run the harness before and after, and report the delta.
- Musical knowledge enters as a prior, never as a filter.
- Prefer a library's own conversions to hand-derived constants.
- Optional dependencies degrade with a warning, never break the run.
- Ask what the benchmark cannot see before trusting what it says.
