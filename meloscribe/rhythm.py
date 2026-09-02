"""Rhythmic plausibility: how metrically believable a transcription is.

Real music places notes on a metrical grid. When a transcription produces
onsets and durations that no human would play - scattered across the beat,
durations unrelated to any note value - that is usually evidence the
*listening* was faulty, not that the performer was strange. This module turns
that intuition into a number: a per-note grid deviation and a per-track
plausibility score, for use as a "check this transcription" flag.

Three deliberate constraints, each of which the obvious implementation gets
wrong:

Nothing here quantises.
    Snapping notes to the grid would erase swing, rubato and melisma, and make
    the output *look* cleaner while being less true - the same mistake as the
    old key filter. The grid observes; it never edits. Two things are fitted
    to the track - the grid's phase offset and whether the beat divides in two
    or in three - and both describe where the *grid* sits, not where any
    individual note sits, so neither can make a scattered performance look
    aligned. Both are priced into the chance level, so the fitting buys no
    free points.

The score is normalised against chance.
    A raw "fraction of notes near a subdivision" is not interpretable: with
    sixteenth-note subdivisions a *uniformly random* onset already lands within
    a plausible tolerance about half the time, so random timing scores ~0.5 and
    the metric looks like it works when it does not. Both components subtract
    their own chance level, so 0.0 means "indistinguishable from random" and
    1.0 means "exactly on the grid".

Tempo is validated before anything is derived from it.
    Beat trackers routinely report double or half the real tempo, and a grid at
    the wrong metrical level makes correct rhythms look implausible (too coarse)
    or makes everything look aligned (too fine). `BeatGrid.confidence` combines
    beat-interval regularity with on-beat versus off-beat onset salience, and a
    grid that fails is marked unreliable rather than silently trusted.

Two things it cannot see, both structural rather than fixable:

    *Under-segmentation.* Two notes run together produce one note that starts
    on the grid and lasts the sum of two grid values, which is usually itself a
    grid value. Merging every second note measured 0.63 against 0.89 for the
    correct transcription - a real drop, but much smaller than the damage, and
    a reference-free measure has no way to miss what was never emitted.

    *Durations that coincide with other note values.* A quarter note released
    at 75% is indistinguishable from a dotted eighth, because that is what it
    is. The duty-cycle fit handles a consistent release; it cannot handle one
    that lands exactly on another ratio.

And the standing caveat: this is a *correlate* of error, not a measure of it.
Rubato, free time and heavy syncopation score badly while being transcribed
perfectly. It belongs as a flag and, at most, a weak prior - never a filter.

Which is why the whole stage is **off by default** (`--rhythm`, or
`TranscriptionRequest.assess_rhythm`). It has been validated against a
synthetic corruption benchmark and exactly one real track, and a diagnostic
resting on one track should not be spending four seconds and two output columns
on runs that did not ask for it. See FLAG_THRESHOLD for what that single track
actually showed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cache import Cache

# Sixteenth notes, for display and for the coarse "on the grid" flag. Finer
# grids do not describe more music, they just make alignment easier to hit by
# accident - and the chance normalisation then has to subtract most of the
# score back off again.
DEFAULT_SUBDIVISIONS = 4

# Where in a beat a note may legally start, as a fraction of the beat. A purely
# binary grid is not a neutral simplification - it cannot express a triplet at
# all, and scored correctly-transcribed triplet material at 0.28 against 0.97
# for the same melody in eighths.
#
# The two divisions are offered as alternatives and the better-fitting one is
# chosen per track, rather than unioned into one six-position grid. The union
# was tried first and is too dense to measure anything: with six positions a
# beat it puts the chance level at 0.85, so a real vocal whose onsets sat a
# median of 18ms from the grid still scored 0.11 - the deviations were small
# and the metric could not tell, because random onsets are nearly as close.
# Choosing one division per track is a single bit of fitting, priced into the
# chance level like the offset, and it buys back the dynamic range.
#
# Positions are shared by every beat; there is no attempt to model a bar line,
# since melody onsets say very little about where the downbeat is.
BEAT_DIVISIONS: Dict[str, Tuple[float, ...]] = {
    'binary': (0.0, 1 / 4, 1 / 2, 3 / 4),
    'ternary': (0.0, 1 / 3, 2 / 3),
}

# Onset tolerance, in beats. 0.08 beat is 40ms at 120bpm - close to mir_eval's
# 50ms onset tolerance, and comfortably wider than the 10ms analysis frame.
#
# Swept over 0.06-0.15 against the corruption benchmark, and the normalised
# scores barely moved (clean 0.854 -> 0.879, scramble 0.159 -> 0.180): a wider
# tolerance raises the chance level by almost exactly as much as it raises the
# raw score, and the normalisation cancels it. That insensitivity is the reason
# to prefer the low end - by 0.15 the chance level is 0.95, so the whole score
# lives in the top 5% of the raw range and small differences get multiplied by
# twenty. 0.08 keeps the arithmetic well conditioned.
ONSET_SIGMA_BEATS = 0.08

# How far the global phase offset may move the grid, in beats. Smaller than the
# closest spacing between grid positions (1/4 to 1/3, one twelfth of a beat) so
# the correction can never re-assign a note to a different position.
OFFSET_LIMIT_BEATS = 0.08

# Duration tolerance in log2 units: 0.10 is roughly +/-7%, chosen against the
# chance level rather than by feel - see _duration_chance.
DURATION_SIGMA_LOG2 = 0.10

# Note values a melody actually uses, as multiples of a beat. Triplets and
# dotted values are included: excluding them would penalise correct
# transcriptions of ordinary music. Values closer together than about a
# tolerance are left out (a dotted sixteenth sits 0.17 in log2 from a triplet
# eighth), since a ratio set dense enough that everything is near something
# measures nothing.
METRICAL_RATIOS = (0.125, 1 / 6, 0.25, 1 / 3, 0.5, 2 / 3, 0.75,
                   1.0, 1.5, 2.0, 3.0, 4.0)

# Durations outside this range are not scored against the ratio set: below it
# everything is close to something, above it a held note is unmeasurable.
DURATION_RANGE_BEATS = (0.1, 6.0)

# Below this the grid is reported but nothing derived from it is trusted.
MIN_TEMPO_CONFIDENCE = 0.45

# Below this the transcription is flagged for review. Deliberately far below
# where the synthetic benchmark alone would put it, and the reason is the only
# real-audio evidence there is.
#
# On one real track, scored in 40-note windows against the same grid, our note
# onsets scored 0.068 +/- 0.083 and an independent onset detector on the same
# vocal stem scored 0.365 +/- 0.138 (range 0.10-0.64). The synthetic-derived
# threshold of 0.35 sat in the *middle* of that baseline's spread: it fired on
# 48% of the windows of a reference that is not even our transcription. A flag
# that is a coin toss on decent input is worse than no flag.
#
# 0.10 is anchored to the worst window that baseline produced on real music, so
# the flag means "below anything a naive onset detector managed here" rather
# than "below what a sequencer would manage". It still catches the real defect
# (our notes never exceeded 0.20 in any window).
#
# This is one track, so it remains provisional in the other direction too: a
# threshold anchored to a single baseline is a floor, not a calibration. Widen
# the evidence before raising it.
FLAG_THRESHOLD = 0.10

# Tempo outside this range is more likely a metrical-level error than a real
# reading, so it is flagged - never silently folded (see _tempo_diagnostics).
PLAUSIBLE_BPM = (55.0, 190.0)

_CACHE_VERSION = 1


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------

@dataclass
class BeatGrid:
    """A tempo, its beat positions, and how much to believe them."""
    bpm: float
    beat_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    subdivisions: int = DEFAULT_SUBDIVISIONS
    confidence: float = 0.0
    # A single global correction to the grid's phase, in beats. Sung melody
    # sits systematically slightly behind the band's beat; correcting that once
    # for a whole track is a property of the grid, not of any note.
    offset: float = 0.0
    # Which division of the beat the melody actually uses. Fitted per track by
    # `analyse`, not assumed - see BEAT_DIVISIONS.
    division: str = 'binary'
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.beat_times = np.asarray(self.beat_times, dtype=float)

    @property
    def reliable(self) -> bool:
        return (self.confidence >= MIN_TEMPO_CONFIDENCE
                and self.beat_times.size >= 4)

    @property
    def beat_period(self) -> float:
        if self.beat_times.size >= 2:
            return float(np.median(np.diff(self.beat_times)))
        return 60.0 / self.bpm if self.bpm > 0 else 0.0

    def beat_positions(self, times) -> np.ndarray:
        """Convert seconds to a continuous position in beats.

        Interpolated through the *tracked* beats rather than extrapolated from
        a constant tempo, so a track that speeds up or slows down keeps a grid
        that follows it instead of drifting out of phase after a minute.
        """
        times = np.asarray(times, dtype=float)
        period = self.beat_period
        if self.beat_times.size < 2 or period <= 0:
            return (times / period if period > 0
                    else np.zeros_like(times)) - self.offset

        index = np.arange(self.beat_times.size, dtype=float)
        positions = np.interp(times, self.beat_times, index)
        # Outside the tracked span, continue at the local tempo rather than
        # clamping: clamping would pile every early or late note onto a single
        # grid point and read as perfect alignment.
        before = times < self.beat_times[0]
        after = times > self.beat_times[-1]
        positions[before] = (times[before] - self.beat_times[0]) / period
        positions[after] = (index[-1]
                            + (times[after] - self.beat_times[-1]) / period)
        return positions - self.offset

    def deviations(self, times) -> np.ndarray:
        """Signed distance from each time to its nearest grid position, in beats.

        Positive means late. The candidate set is `BEAT_POSITIONS` plus the
        next beat's downbeat, so a note that anticipates the beat is measured
        as slightly early rather than as three quarters of a beat late.
        """
        positions = np.atleast_1d(self.beat_positions(times))
        return _grid_deviations(positions - np.floor(positions),
                                division=self.division)

    def subdivision_times(self) -> np.ndarray:
        """Every subdivision instant across the tracked span, in seconds."""
        if self.beat_times.size < 2:
            return self.beat_times.copy()
        steps = np.arange((self.beat_times.size - 1) * self.subdivisions + 1)
        index = steps / self.subdivisions
        return np.interp(index, np.arange(self.beat_times.size),
                         self.beat_times)

    def to_dict(self) -> Dict[str, Any]:
        return {'bpm': round(self.bpm, 2),
                'beat_times': [round(float(t), 4) for t in self.beat_times],
                'subdivisions': self.subdivisions,
                'confidence': round(self.confidence, 4),
                'offset': round(self.offset, 4),
                'division': self.division,
                'reliable': self.reliable,
                'warnings': list(self.warnings)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeatGrid':
        return cls(bpm=float(data['bpm']),
                   beat_times=np.asarray(data.get('beat_times', []),
                                         dtype=float),
                   subdivisions=int(data.get('subdivisions',
                                             DEFAULT_SUBDIVISIONS)),
                   confidence=float(data.get('confidence', 0.0)),
                   offset=float(data.get('offset', 0.0)),
                   division=str(data.get('division', 'binary')),
                   warnings=list(data.get('warnings', [])))


def constant_tempo_grid(bpm: float, duration: float, start: float = 0.0,
                        subdivisions: int = DEFAULT_SUBDIVISIONS,
                        confidence: float = 1.0) -> BeatGrid:
    """A perfectly regular grid, for tests and for known-tempo material."""
    period = 60.0 / bpm
    n_beats = max(2, int(np.floor((duration - start) / period)) + 1)
    return BeatGrid(bpm=bpm, beat_times=start + np.arange(n_beats) * period,
                    subdivisions=subdivisions, confidence=confidence)


# --------------------------------------------------------------------------
# Tempo detection
# --------------------------------------------------------------------------

def detect_grid(audio_path, subdivisions: int = DEFAULT_SUBDIVISIONS,
                cache: Optional[Cache] = None,
                force: bool = False) -> BeatGrid:
    """Track tempo and beats in an audio file.

    Give this the **full mix**, not the isolated vocal. Beat tracking keys off
    percussive onsets, and a separated vocal has had exactly those removed - a
    grid tracked from the vocal stem would describe the singer's own phrasing,
    which is the very thing we want an independent reference for.
    """
    audio_path = Path(audio_path)
    params = {'subdivisions': subdivisions, 'v': _CACHE_VERSION}
    cache = cache or Cache()
    entry = cache.entry(audio_path, 'rhythm', params, expected=['grid.json'])

    if entry.hit and not force:
        try:
            return BeatGrid.from_dict(json.loads(
                (entry.path / 'grid.json').read_text(encoding='utf-8')))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass  # a corrupt entry is a miss, not a failure

    grid = _track_beats(audio_path, subdivisions)

    entry.path.mkdir(parents=True, exist_ok=True)
    (entry.path / 'grid.json').write_text(json.dumps(grid.to_dict()),
                                          encoding='utf-8')
    entry.write_meta({'stage': 'rhythm', 'source': str(audio_path), **params})
    return grid


def _track_beats(audio_path: Path, subdivisions: int) -> BeatGrid:
    try:
        import librosa
    except ImportError:
        return BeatGrid(bpm=0.0, subdivisions=subdivisions, confidence=0.0,
                        warnings=['librosa is not installed; '
                                  'no rhythmic analysis'])

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if y.size < sr:
        return BeatGrid(bpm=0.0, subdivisions=subdivisions, confidence=0.0,
                        warnings=['audio is too short for beat tracking'])

    envelope = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=envelope,
                                                 sr=sr, units='frames')
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = np.asarray(librosa.frames_to_time(beat_frames, sr=sr),
                            dtype=float)

    confidence, warnings = _tempo_diagnostics(envelope, beat_frames, bpm)
    return BeatGrid(bpm=bpm, beat_times=beat_times, subdivisions=subdivisions,
                    confidence=confidence, warnings=warnings)


def _tempo_diagnostics(envelope: np.ndarray, beat_frames: np.ndarray,
                       bpm: float) -> Tuple[float, List[str]]:
    """How much to believe this tempo, and what is suspicious about it.

    Two independent checks, because they catch different failures:

    *Regularity* - the spread of the beat intervals. A tracker that is guessing
    produces beats that are not evenly spaced, and every grid position derived
    from them inherits that error.

    *Metrical level* - whether the onset envelope agrees that these are the
    beats. If the midpoints between beats are as strong as the beats
    themselves, the real pulse may be twice as fast; if every other beat is
    near-empty, it may be half as fast. Neither is proof - a stream of eighth
    notes looks exactly like the first case - so this lowers confidence and
    raises a warning rather than "correcting" the tempo. Halving a tempo that
    was right is worse than reporting one that is uncertain.
    """
    warnings: List[str] = []
    beat_frames = np.asarray(beat_frames, dtype=float)
    if beat_frames.size < 4:
        return 0.0, ['too few beats detected to form a grid']

    intervals = np.diff(beat_frames)
    mean_interval = float(np.mean(intervals))
    if mean_interval <= 0:
        return 0.0, ['degenerate beat spacing']

    spread = float(np.std(intervals)) / mean_interval
    # 15% jitter is about where a tracked grid stops being usable as a ruler.
    # Some slack is warranted because positions are interpolated through the
    # tracked beats rather than laid out from a constant tempo, so per-beat
    # wobble is largely absorbed rather than accumulated.
    regularity = float(np.clip(1.0 - spread / 0.15, 0.0, 1.0))
    if regularity < 0.5:
        warnings.append(f"beat spacing is uneven ({spread:.0%} jitter); the "
                        f"tempo estimate is probably wrong")

    salience = 1.0
    onbeat = _sample_envelope(envelope, beat_frames)
    offbeat = _sample_envelope(envelope, beat_frames[:-1] + intervals / 2.0)
    floor = float(np.mean(envelope))

    if onbeat > floor:
        if offbeat / onbeat > 0.95:
            warnings.append('off-beats are as strong as the beats; the tempo '
                            'may be half the real one')
            salience *= 0.7
        # Alternate-beat asymmetry: the signature of a doubled tempo.
        even = _sample_envelope(envelope, beat_frames[0::2])
        odd = _sample_envelope(envelope, beat_frames[1::2])
        if min(even, odd) / max(even, odd, 1e-9) < 0.5:
            warnings.append('every other beat is near-empty; the tempo may be '
                            'double the real one')
            salience *= 0.7
    else:
        warnings.append('there is no onset energy at the detected beats')
        salience *= 0.4

    if not PLAUSIBLE_BPM[0] <= bpm <= PLAUSIBLE_BPM[1]:
        warnings.append(f"{bpm:.0f} BPM is outside the usual range; this is "
                        f"often a metrical-level error")
        salience *= 0.6

    return float(np.clip(regularity * salience, 0.0, 1.0)), warnings


def _sample_envelope(envelope: np.ndarray, frames: np.ndarray) -> float:
    """Mean onset strength at (possibly fractional) frame positions."""
    frames = np.asarray(frames, dtype=float)
    if frames.size == 0 or envelope.size == 0:
        return 0.0
    index = np.clip(np.round(frames).astype(int), 0, envelope.size - 1)
    return float(np.mean(envelope[index]))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _gaussian(distance, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (np.asarray(distance, dtype=float) / sigma) ** 2)


def _grid_deviations(fractions, offset: float = 0.0,
                     division: str = 'binary') -> np.ndarray:
    """Signed distance from each in-beat position to its nearest grid position."""
    shifted = (np.atleast_1d(np.asarray(fractions, dtype=float)) - offset) % 1.0
    candidates = np.array(BEAT_DIVISIONS[division] + (1.0,), dtype=float)
    offsets = shifted[:, None] - candidates[None, :]
    nearest = np.argmin(np.abs(offsets), axis=1)
    return offsets[np.arange(offsets.shape[0]), nearest]


def _fit_grid(fractions, sigma: float) -> Tuple[str, float, float]:
    """Choose the division and phase that best explain these onsets.

    Returns `(division, offset, raw_score)`. A direct search rather than a
    circular mean: the grid positions are not evenly spaced, and a circular
    statistic assumes they are, which made it read a consistent 50ms lag on
    triplet material as no lag at all.

    The phase search range is deliberately narrower than the smallest gap
    between grid positions, so it can only correct where the grid sits; it can
    never slide a note onto a *different* position than the one it was near.
    """
    trials = np.linspace(-OFFSET_LIMIT_BEATS, OFFSET_LIMIT_BEATS, 41)
    best = ('binary', 0.0, -1.0)
    for division in BEAT_DIVISIONS:
        for offset in trials:
            score = float(np.mean(_gaussian(
                _grid_deviations(fractions, offset, division), sigma)))
            if score > best[2]:
                best = (division, float(offset), score)
    return best


@lru_cache(maxsize=64)
def _onset_chance(n_notes: int, sigma: float) -> float:
    """Expected onset score for `n_notes` onsets placed uniformly at random.

    Computed rather than assumed, because it is large: against a binary
    division with a 40ms tolerance, random onsets already score around 0.6 on
    the raw measure. Without subtracting this, "the notes are near the grid"
    would be just as true of noise.

    The simulation runs the *whole* estimator - phase offset and division
    choice included - which is the point: fitting those to n onsets improves a
    random note list too, and a chance level that ignored it would hand every
    short track free points. It therefore depends on the note count, and is
    lower for a long track than a short one.
    """
    rng = np.random.default_rng(20240917)
    trials = 200 if n_notes <= 64 else 60
    total = 0.0
    for _ in range(trials):
        total += _fit_grid(rng.random(max(1, n_notes)), sigma)[2]
    return total / trials


@lru_cache(maxsize=64)
def _duration_chance(n_notes: int, sigma: float) -> float:
    """Expected duration score for `n_notes` durations drawn at random.

    Log-uniform, not uniform: the ratio set is itself geometric, so a uniform
    draw would over-sample the sparse long end and understate how easy the
    measure is to satisfy by accident.

    Like the onset chance, this simulates the whole estimator including the
    duty fit, for the same reason: fitting one scalar to a random duration list
    also improves it, and leaving that out let a scrambled note list score 0.4
    on material where it should have scored nothing.
    """
    low, high = DURATION_RANGE_BEATS
    rng = np.random.default_rng(20240918)
    trials = 300 if n_notes <= 64 else 100
    total = 0.0
    for _ in range(trials):
        samples = np.exp(rng.uniform(np.log(low), np.log(high),
                                     max(1, n_notes)))
        total += float(np.mean(_duration_scores(
            samples, sigma, _estimate_duty(samples, sigma))))
    return total / trials


def _duration_log_error(durations) -> np.ndarray:
    """Signed log2 distance from each duration (in beats) to its nearest ratio.

    Measured in log2 units so the tolerance is proportional: a sixteenth note
    may be 7% short and so may a whole note, which is how performed timing
    error actually scales.
    """
    durations = np.atleast_1d(np.asarray(durations, dtype=float))
    ratios = np.asarray(METRICAL_RATIOS, dtype=float)
    error = np.log2(np.maximum(durations, 1e-6)[:, None] / ratios[None, :])
    nearest = np.argmin(np.abs(error), axis=1)
    return error[np.arange(error.shape[0]), nearest]


def _duration_scores(durations, sigma: float,
                     duty: float = 0.0) -> np.ndarray:
    """Score each duration against the nearest metrical ratio.

    `duty` is a single global log2 correction, the duration counterpart of the
    grid's phase offset. Both a real singer and our own segmentation release
    notes early - a note written as a quarter sounds for rather less than one -
    so without it a perfectly transcribed melody is scored as systematically
    wrong. It shifts all durations together, so it cannot make scattered
    durations look metrical.
    """
    return _gaussian(_duration_log_error(durations) - duty, sigma)


def _estimate_duty(durations, sigma: float) -> float:
    """The track's systematic shortening, in log2, or 0 if there isn't one.

    Capped at a factor of two short and never allowed to lengthen: a
    consistently clipped release is a real property of singing, whereas a
    correction that stretched every note would just be fitting the ratio set to
    whatever came in.
    """
    durations = np.atleast_1d(np.asarray(durations, dtype=float))
    if durations.size < MIN_NOTES_FOR_SCORE:
        return 0.0
    error = _duration_log_error(durations)
    duty = float(np.median(error))
    # Only correct a lag the durations agree on; a scattered set has a median
    # that means nothing, so require it to be small relative to the spread.
    if float(np.median(np.abs(error - duty))) > 2.0 * sigma:
        return 0.0
    return float(np.clip(duty, -1.0, 0.0))


def _normalise(raw: float, chance: float) -> float:
    """Rescale a raw score so chance maps to 0 and perfection to 1."""
    if chance >= 1.0:
        return 0.0
    return float(np.clip((raw - chance) / (1.0 - chance), 0.0, 1.0))


@dataclass
class NoteRhythm:
    """The rhythmic view of one note."""
    # Signed distance to the nearest subdivision, in beats. Positive is late.
    deviation_beats: float
    deviation_s: float
    onset_score: float          # 0-1, chance-normalised
    duration_beats: float
    duration_score: float       # 0-1, chance-normalised
    # Convenience for the UI: within half a subdivision's tolerance window.
    on_grid: bool

    def to_dict(self) -> Dict[str, Any]:
        return {'deviation_beats': round(self.deviation_beats, 4),
                'deviation_s': round(self.deviation_s, 4),
                'onset_score': round(self.onset_score, 4),
                'duration_beats': round(self.duration_beats, 3),
                'duration_score': round(self.duration_score, 4),
                'on_grid': self.on_grid}


@dataclass
class RhythmReport:
    """Per-note rhythmic detail plus the track-level plausibility score."""
    grid: BeatGrid
    notes: List[NoteRhythm] = field(default_factory=list)
    onset_score: float = 0.0
    duration_score: float = 0.0
    plausibility: float = 0.0
    # False when the tempo could not be trusted, or there were too few notes.
    reliable: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def on_grid_fraction(self) -> float:
        if not self.notes:
            return 0.0
        return float(np.mean([n.on_grid for n in self.notes]))

    @property
    def flagged(self) -> bool:
        """Whether this transcription is worth a second look.

        Only ever true when the grid itself was trustworthy - an unreliable
        tempo produces a low score for reasons that have nothing to do with the
        transcription, and flagging on that would train the user to ignore it.
        """
        return self.reliable and self.plausibility < FLAG_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {'grid': self.grid.to_dict(),
                'onset_score': round(self.onset_score, 4),
                'duration_score': round(self.duration_score, 4),
                'plausibility': round(self.plausibility, 4),
                'on_grid_fraction': round(self.on_grid_fraction, 4),
                'reliable': self.reliable,
                'flagged': self.flagged,
                'warnings': list(self.warnings)}

    def summary(self) -> str:
        if not self.reliable:
            reason = self.warnings[0] if self.warnings else 'no reliable tempo'
            return f"rhythm: not assessed ({reason})"
        return (f"rhythm: {self.plausibility:.2f} plausibility at "
                f"{self.grid.bpm:.0f} BPM, {self.grid.division} division "
                f"({self.on_grid_fraction:.0%} of notes on the grid)"
                + ('  [check this transcription]' if self.flagged else ''))


# Fewer notes than this and the score is noise: a handful of notes can look
# metrical or unmetrical by luck.
MIN_NOTES_FOR_SCORE = 8


def analyse(notes: Sequence, grid: BeatGrid,
            estimate_offset: bool = True) -> RhythmReport:
    """Score a note list against a beat grid.

    `notes` needs `.start` and `.duration` (a `TranscribedNote`) or `.onset`
    and `.duration` (an eval `Note`); both are accepted so the harness can
    score ground truth without converting it first.
    """
    report = RhythmReport(grid=grid, warnings=list(grid.warnings))
    if not notes:
        report.warnings.append('no notes to assess')
        return report

    onsets = np.array([_onset_of(n) for n in notes], dtype=float)
    durations = np.array([float(n.duration) for n in notes], dtype=float)

    grid.offset = 0.0
    positions = grid.beat_positions(onsets)
    fractions = positions - np.floor(positions)
    if estimate_offset and grid.beat_times.size >= 2:
        grid.division, grid.offset, _ = _fit_grid(fractions,
                                                  ONSET_SIGMA_BEATS)

    deviations = _grid_deviations(fractions, grid.offset, grid.division)
    period = grid.beat_period

    onset_chance = _onset_chance(len(notes), ONSET_SIGMA_BEATS)
    raw_onset = _gaussian(deviations, ONSET_SIGMA_BEATS)

    duration_beats = durations / period if period > 0 else np.zeros_like(durations)
    duty = _estimate_duty(duration_beats, DURATION_SIGMA_LOG2)
    raw_duration = _duration_scores(duration_beats, DURATION_SIGMA_LOG2, duty)
    # Durations we declared unmeasurable are excluded rather than scored as
    # failures: a two-bar held note is not evidence of anything.
    measurable = ((duration_beats >= DURATION_RANGE_BEATS[0])
                  & (duration_beats <= DURATION_RANGE_BEATS[1]))
    duration_chance = _duration_chance(int(np.sum(measurable)),
                                       DURATION_SIGMA_LOG2)

    # "On the grid" is judged against the chosen division's own spacing, so a
    # ternary track is not held to a sixteenth-note ruler it never used.
    tolerance = 0.5 / len(BEAT_DIVISIONS[grid.division])
    report.notes = [
        NoteRhythm(deviation_beats=float(deviations[i]),
                   deviation_s=float(deviations[i] * period),
                   onset_score=_normalise(float(raw_onset[i]), onset_chance),
                   duration_beats=float(duration_beats[i]),
                   duration_score=(_normalise(float(raw_duration[i]),
                                              duration_chance)
                                   if measurable[i] else float('nan')),
                   on_grid=bool(abs(deviations[i]) <= tolerance * 0.5))
        for i in range(len(notes))]

    report.onset_score = _normalise(float(np.mean(raw_onset)), onset_chance)
    report.duration_score = (
        _normalise(float(np.mean(raw_duration[measurable])), duration_chance)
        if np.any(measurable) else 0.0)

    # Onsets are weighted more heavily than durations because note-off timing
    # in sung melody is genuinely ambiguous - the same reason the note metrics
    # score onsets only. Durations still carry real signal about shredding, so
    # they are not dropped, just discounted.
    report.plausibility = 0.65 * report.onset_score + 0.35 * report.duration_score

    if not grid.reliable:
        report.reliable = False
        if not report.warnings:
            report.warnings.append('tempo estimate is not reliable')
    elif len(notes) < MIN_NOTES_FOR_SCORE:
        report.reliable = False
        report.warnings.append(
            f"only {len(notes)} notes; too few to assess rhythm")
    else:
        report.reliable = True

    return report


def _onset_of(note) -> float:
    """The start time of a note, whichever vocabulary it uses."""
    for attribute in ('start', 'onset'):
        value = getattr(note, attribute, None)
        if value is not None:
            return float(value)
    raise AttributeError(f"{note!r} has neither .start nor .onset")


def assess(notes: Sequence, audio_path, subdivisions: int = DEFAULT_SUBDIVISIONS,
           cache: Optional[Cache] = None, force: bool = False) -> RhythmReport:
    """Detect the grid from audio and score `notes` against it."""
    grid = detect_grid(audio_path, subdivisions=subdivisions, cache=cache,
                       force=force)
    return analyse(notes, grid)
