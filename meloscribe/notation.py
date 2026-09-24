"""From notes in seconds to notes in bars and beats, for sheet music.

Every other export keeps the transcription's own timing, which is the honest
thing to do with a performance - and which is also why a notation program
opens our MIDI as 64th-note ties across every beat. Notation needs note values,
so this module quantises, in a separate step and a separate module: the rhythm
module's rule that the grid observes and never edits still holds for every
other output, and nothing here writes back to a note.

The pieces, and why each is shaped the way it is:

The grid is regularised before anything snaps to it.
    `rhythm.detect_grid` interpolates through the tracker's beats, which is the
    right ruler for a deviation score but a poor one to snap to: on the
    metrical benchmark the tracked beats sit 10-37ms behind the pulse and
    wobble around it by up to 178ms, because a melody note pulls the tracker
    off the click. A local linear fit through +/-8 beats keeps a drifting
    tempo and removes most of the wobble; `rhythm.analyse` then fits the
    track's phase offset and binary/ternary feel, both of which describe the
    grid rather than any note (see rhythm.py).

Sixteenths by default, triplet eighths per beat only on clear evidence.
    Each beat independently chooses between a sixteenth and a triplet-eighth
    grid by the likelihood of its onsets under each, against a prior that
    depends on the whole track's feel: a straight track needs strong evidence
    before a beat turns into triplets, because a spurious triplet in a straight
    song is much harder to read than a missed one written as sixteenths.

Onsets are assigned jointly, not snapped one by one.
    Two onsets that round to the same slot would make a zero-length note. A
    small dynamic programme gives every note its own slot at the least total
    displacement, which is also what makes the minimum note value (one grid
    step) hold without deleting anything: quantising is a prior on where notes
    go, never a filter on which notes exist.

Ends are soft.
    Sung notes stop early, so a note released 92% of the way to the next one is
    legato, not a note plus a 64th rest. Gaps shorter than ABSORB_REST_BEATS are
    absorbed into the note before them; longer ones are real rests, whose start
    is corrected by the track's own release - how much of its written length a
    legato note sounds for - before it snaps.

What this does not know, and says so:
    rhythm.py exposes no downbeat and no meter - its grid is a pulse with no
    bar lines - so the time signature is 4/4 and bar 1 starts on the beat of
    the first note unless the caller states otherwise (`beats_per_bar`,
    `pickup`). A downbeat heuristic on accents was not added: a pop backbeat
    puts the loudest onsets on beats 2 and 4, and there is no real annotated
    audio in the repo to show a heuristic gets past that. When the grid itself
    is unreliable, a fixed tempo from the median inter-onset interval stands in
    and the score is marked approximate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from . import rhythm as rhythm_mod
from .rhythm import BeatGrid

# Divisions of a beat: the least common multiple of 4 (sixteenths) and 3
# (triplet eighths), so both grids are exact integers.
TICKS_PER_BEAT = 12

# Where a note may start or end inside a beat, in ticks, for each division.
POSITIONS: Dict[str, Tuple[int, ...]] = {
    'binary': (0, 3, 6, 9),   # sixteenths
    'ternary': (0, 4, 8),     # triplet eighths
}

# The onset timing error assumed when weighing a triplet against sixteenths,
# in seconds. Taken from `rhythm_eval`'s 'tight' corruption - the error a
# *correct* transcription still has - rather than chosen to make a number go
# up. Converted to beats per track, so evidence is weighed at its real tempo.
ONSET_JITTER_S = 0.025

# Log prior odds for sixteenths over triplets in one beat, by the track's
# overall feel as `rhythm.analyse` fits it. In a straight track a beat needs
# about as much evidence as two clean triplet onsets provide at 120 BPM before
# it becomes triplets; in a track that swings or shuffles, the beat's own
# onsets decide.
TRIPLET_PRIOR = {'binary': 2.0, 'ternary': 0.0}

# A silence shorter than this, in beats, is articulation rather than a rest:
# it is absorbed into the note before it. Between a sixteenth rest (0.25) and
# an eighth rest (0.5), so an eighth rest survives +/-30ms of jitter at 120 BPM
# while the gap after a legato whole note (8% of four beats) does not.
ABSORB_REST_BEATS = 0.375

# How far either side of a beat the grid regularisation looks, in beats.
SMOOTHING_HALF_WINDOW = 8

# The fallback tempo from note spacing is folded into this range by octaves.
# One octave wide, so the fold always has exactly one answer.
FALLBACK_BPM_RANGE = (70.0, 140.0)

# Note values written as one symbol when a note starts on a beat, longest
# first, in ticks; see `_on_beat_length` for where each may start.
_WHOLE, _DOTTED_HALF, _HALF, _DOTTED_QUARTER = 48, 36, 24, 18

# (type, dots) for each length in ticks, and for the lengths that only exist
# as triplets.
_VALUES = {3: ('16th', 0), 6: ('eighth', 0), 9: ('eighth', 1),
           12: ('quarter', 0), 18: ('quarter', 1), 24: ('half', 0),
           36: ('half', 1), 48: ('whole', 0)}
_TRIPLET_VALUES = {4: ('eighth', 0), 8: ('quarter', 0)}


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class LyricMark:
    """What to print under (or at) one note."""
    text: str
    # 'word': one word under the first note it is sung on (word timings).
    # 'line': a whole lyric line at its first note (line timings only).
    kind: str = 'word'
    # The word is held over the notes that follow it: a melisma.
    extend: bool = False


@dataclass
class QuantizedNote:
    """One note on the grid. Times are ticks from the start of the score."""
    source: int              # index in the note list that was quantised
    start: int
    end: int
    midi: int
    name: str                # spelled as the note itself spells it
    confidence: float = 1.0
    lyric: Optional[LyricMark] = None

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class Piece:
    """One printed symbol: a note or rest of a single value, maybe tied."""
    start: int
    duration: int
    note: Optional[QuantizedNote] = None   # None is a rest
    type: str = 'quarter'
    dots: int = 0
    triplet: bool = False
    tie_start: bool = False
    tie_stop: bool = False
    tuplet_start: bool = False
    tuplet_stop: bool = False
    measure_rest: bool = False

    @property
    def is_rest(self) -> bool:
        return self.note is None

    @property
    def first_of_note(self) -> bool:
        """The piece a note's lyric and colour belong to."""
        return self.note is not None and not self.tie_stop


@dataclass
class Measure:
    number: int              # 0 for a pickup, then 1, 2, ...
    start: int               # ticks from the start of the score
    length: int
    pieces: List[Piece] = field(default_factory=list)

    @property
    def implicit(self) -> bool:
        """A pickup: shorter than the time signature, and not counted."""
        return self.number == 0


@dataclass
class QuantizedScore:
    """A melody laid out in beats, ready to write as notation."""
    notes: List[QuantizedNote]
    bpm: float
    beats_per_bar: int = 4
    pickup: int = 0          # ticks in the pickup measure; 0 for none
    # Beats (counted from the start of the score) written as triplet eighths.
    ternary_beats: FrozenSet[int] = frozenset()
    # True when no trustworthy beat grid was found and the tempo is a guess.
    approximate: bool = False
    # Where the start of the score falls in the recording, in seconds.
    origin_s: float = 0.0
    grid_source: str = 'tracked'
    warnings: List[str] = field(default_factory=list)

    ticks_per_beat: int = TICKS_PER_BEAT

    @property
    def bar_ticks(self) -> int:
        return self.beats_per_bar * TICKS_PER_BEAT

    @property
    def end(self) -> int:
        """The end of the last measure, in ticks."""
        last = max((n.end for n in self.notes), default=0)
        if last <= self.pickup:
            return self.pickup if self.pickup else self.bar_ticks
        bars = math.ceil((last - self.pickup) / self.bar_ticks)
        return self.pickup + bars * self.bar_ticks

    def seconds(self, tick: float) -> float:
        """A tick as seconds from the start of the score, at the fixed tempo."""
        return tick / TICKS_PER_BEAT * 60.0 / self.bpm

    def measures(self) -> List[Measure]:
        return layout(self)


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------

def regularise(grid: BeatGrid,
               half_window: int = SMOOTHING_HALF_WINDOW) -> BeatGrid:
    """A copy of `grid` with its beats smoothed onto a locally steady tempo.

    Beats are first re-indexed against the median period, so a beat the
    tracker skipped leaves a gap to fill rather than silently merging two
    bars' worth of tempo, and a doubled beat is dropped. Each beat is then
    replaced by a straight line fitted through its neighbours: a song that
    drifts keeps its drift, a tracker that wobbles loses its wobble.
    """
    beats = np.asarray(grid.beat_times, dtype=float)
    out = BeatGrid(bpm=grid.bpm, beat_times=beats.copy(),
                   subdivisions=grid.subdivisions, confidence=grid.confidence,
                   offset=0.0, division=grid.division,
                   warnings=list(grid.warnings))
    if beats.size < 4:
        return out

    period = float(np.median(np.diff(beats)))
    if period <= 0:
        return out
    index, kept = [0], [beats[0]]
    for t in beats[1:]:
        step = int(round((t - kept[-1]) / period))
        if step < 1:
            continue  # a doubled beat
        index.append(index[-1] + step)
        kept.append(t)
    index_arr = np.asarray(index, dtype=float)
    kept_arr = np.asarray(kept, dtype=float)

    every = np.arange(index[-1] + 1, dtype=float)
    smoothed = np.empty_like(every)
    for i, position in enumerate(every):
        near = np.abs(index_arr - position) <= half_window
        if np.count_nonzero(near) < 2:
            near = np.argsort(np.abs(index_arr - position))[:2]
        slope, intercept = np.polyfit(index_arr[near], kept_arr[near], 1)
        smoothed[i] = slope * position + intercept

    # A fit can only cross over itself on a wildly irregular grid, which the
    # reliability gate should already have refused; keep the raw beats then.
    if np.any(np.diff(smoothed) <= 0):
        return out
    out.beat_times = smoothed
    out.bpm = 60.0 / float(np.median(np.diff(smoothed)))
    return out


def grid_for_output(output, audio_path=None,
                    cache=None) -> Tuple[Optional[BeatGrid], List[str]]:
    """The beat grid a finished transcription should be written against, and
    what went wrong finding one.

    The rhythm stage's own grid when it ran; otherwise one tracked now from
    the source audio - the full mix, which is what `rhythm.detect_grid` needs
    and caches, so exporting twice tracks once. None when there is no audio
    or tracking fails, and `quantize` then falls back to the note spacing.
    """
    report = getattr(output, 'rhythm', None)
    if report is not None and report.grid is not None:
        return report.grid, []

    request = getattr(output, 'request', None)
    path = audio_path or (request.input_path if request is not None else None)
    if path is None:
        return None, ['no source audio to track the beat from']
    try:
        grid = rhythm_mod.detect_grid(path, cache=cache)
    except Exception as exc:   # a broken or missing file must not sink export
        return None, [f"beat tracking failed ({exc})"]
    warnings: List[str] = []
    if request is not None and getattr(request, 'vocals_only', False):
        warnings.append('the beat was tracked from an isolated vocal, which '
                        'has no percussion to lock onto')
    return grid, warnings


def fallback_grid(starts: Sequence[float],
                  bpm_range: Tuple[float, float] = FALLBACK_BPM_RANGE
                  ) -> BeatGrid:
    """A fixed-tempo grid from the note spacing alone, starting on the first
    note. Only a guess: the median gap between onsets is taken to be a note
    value a power of two away from the beat, which is often true and never
    checked. `confidence` is 0 so nothing mistakes it for a tracked grid."""
    starts = np.sort(np.asarray(starts, dtype=float))
    gaps = np.diff(starts)
    gaps = gaps[gaps > 0.05]
    bpm = 120.0
    if gaps.size:
        bpm = 60.0 / float(np.median(gaps))
        low, high = bpm_range
        while bpm < low:
            bpm *= 2.0
        while bpm >= high:
            bpm /= 2.0
    first = float(starts[0]) if starts.size else 0.0
    span = (float(starts[-1]) - first if starts.size else 0.0) + 8 * 60.0 / bpm
    grid = rhythm_mod.constant_tempo_grid(bpm, span + first, start=first,
                                          confidence=0.0)
    grid.warnings = ['no reliable beat grid; tempo estimated from the '
                     'spacing of the notes']
    return grid


# --------------------------------------------------------------------------
# Quantising
# --------------------------------------------------------------------------

def quantize(notes: Sequence, grid: Optional[BeatGrid] = None,
             beats_per_bar: int = 4, pickup: int = 0,
             lyrics=None) -> QuantizedScore:
    """Lay `notes` out on a metrical grid.

    `notes` need start/end times (`TranscribedNote`) or onset/offset (the
    eval `Note`). `grid` is a tracked beat grid; without one, or with one the
    tracker itself does not trust, the tempo is estimated from the notes and
    the result is marked approximate. `beats_per_bar` is 4 or 3, and `pickup`
    is the number of beats before the first bar line, counted from the beat
    the first note starts in - neither is detected (see the module docstring).
    `lyrics` is the `TimedLyrics` the notes were labelled from, if any.
    """
    if beats_per_bar not in (2, 3, 4):
        raise ValueError(f"beats_per_bar must be 2, 3 or 4, not {beats_per_bar}")
    order = sorted(range(len(notes)), key=lambda i: _times(notes[i]))
    ordered = [notes[i] for i in order]
    times = np.array([_times(n) for n in ordered], dtype=float).reshape(-1, 2)
    warnings: List[str] = []

    approximate = grid is None or not grid.reliable
    if approximate:
        if grid is not None:
            warnings.extend(grid.warnings or ['beat grid is not reliable'])
        working = fallback_grid(times[:, 0])
        warnings.extend(working.warnings)
        source = 'fallback'
    else:
        working = regularise(grid)
        source = 'tracked'

    if not len(ordered):
        return QuantizedScore(notes=[], bpm=round(working.bpm or 120.0),
                              beats_per_bar=beats_per_bar,
                              approximate=approximate, grid_source=source,
                              warnings=warnings)

    # The phase offset and the track's binary/ternary feel, fitted by the
    # rhythm module on the grid we will snap to. Both move the grid, never a
    # note, and the offset is bounded well inside a sixteenth.
    if working.beat_times.size >= 2:
        rhythm_mod.analyse(ordered, working)
    period = working.beat_period or 60.0 / (working.bpm or 120.0)

    raw_on = working.beat_positions(times[:, 0]) * TICKS_PER_BEAT
    raw_end = working.beat_positions(times[:, 1]) * TICKS_PER_BEAT

    sigma = ONSET_JITTER_S / period if period > 0 else 0.05
    ternary = choose_subdivisions(raw_on / TICKS_PER_BEAT, sigma=sigma,
                                  prior=TRIPLET_PRIOR[working.division])
    onsets = assign_onsets(raw_on, ternary)
    release = _release_factor(raw_on, raw_end)
    ends = assign_ends(onsets, raw_on, raw_end, ternary, release=release)

    # The score starts on the beat the first note is in.
    origin_beat = int(onsets[0] // TICKS_PER_BEAT)
    origin = origin_beat * TICKS_PER_BEAT
    origin_s = _time_at(working, origin_beat)

    marks = lyric_marks(ordered, lyrics)
    placed = [QuantizedNote(source=order[i], start=int(onsets[i] - origin),
                            end=int(ends[i] - origin),
                            midi=int(round(float(_midi(n)))),
                            name=_name(n),
                            confidence=float(getattr(n, 'confidence', 1.0)),
                            lyric=marks[i])
              for i, n in enumerate(ordered)]

    return QuantizedScore(
        notes=placed, bpm=float(60.0 / period) if period > 0 else 120.0,
        beats_per_bar=beats_per_bar,
        pickup=(int(pickup) % beats_per_bar) * TICKS_PER_BEAT,
        ternary_beats=frozenset(b - origin_beat for b in ternary),
        approximate=approximate, origin_s=origin_s, grid_source=source,
        warnings=warnings)


def choose_subdivisions(positions: np.ndarray, sigma: float,
                        prior: float) -> FrozenSet[int]:
    """The beats (by index) whose onsets are better explained by triplets.

    Per beat, the log-likelihood ratio of its onsets under a triplet-eighth
    grid against a sixteenth grid, each onset taken to sit at its nearest
    position with Gaussian timing error `sigma` (in beats). A beat goes
    ternary only when that ratio beats `prior`. Onsets on or next to the beat
    count the same under both, so they neither help nor hurt.
    """
    positions = np.asarray(positions, dtype=float)
    beats = np.floor(positions).astype(int)
    fractions = positions - beats
    binary = np.array((0.0, 0.25, 0.5, 0.75, 1.0))
    ternary = np.array((0.0, 1 / 3, 2 / 3, 1.0))
    d_bin = np.min(np.abs(fractions[:, None] - binary[None, :]), axis=1)
    d_ter = np.min(np.abs(fractions[:, None] - ternary[None, :]), axis=1)
    gain = (d_bin ** 2 - d_ter ** 2) / (2.0 * max(sigma, 1e-3) ** 2)

    chosen = set()
    for beat in np.unique(beats):
        if float(np.sum(gain[beats == beat])) > prior:
            chosen.add(int(beat))
    return frozenset(chosen)


def grid_points(low: float, high: float,
                ternary: FrozenSet[int]) -> np.ndarray:
    """Every grid position in ticks within [low, high], each beat using its
    own division."""
    points: List[int] = []
    for beat in range(int(math.floor(low / TICKS_PER_BEAT)),
                      int(math.floor(high / TICKS_PER_BEAT)) + 1):
        base = beat * TICKS_PER_BEAT
        for offset in POSITIONS['ternary' if beat in ternary else 'binary']:
            if low <= base + offset <= high:
                points.append(base + offset)
    return np.asarray(points, dtype=float)


def assign_onsets(raw: np.ndarray, ternary: FrozenSet[int]) -> np.ndarray:
    """Give every onset its own grid slot, in order, at the least total
    squared displacement.

    The candidates for each onset are the grid points within a beat of it,
    widened only if a dense run cannot otherwise fit. Strictly increasing
    slots are what make the minimum note value hold: a note can be moved, and
    lengthened to one grid step, but it is never merged away or dropped.
    """
    raw = np.asarray(raw, dtype=float)
    window = float(TICKS_PER_BEAT)
    for _ in range(8):
        candidates = [grid_points(r - window, r + window, ternary) for r in raw]
        chosen = _monotone_assignment(raw, candidates)
        if chosen is not None:
            return chosen
        window *= 2.0
    raise RuntimeError('could not place the onsets on the grid')


def _monotone_assignment(raw: np.ndarray,
                         candidates: List[np.ndarray]) -> Optional[np.ndarray]:
    totals: List[np.ndarray] = []
    links: List[np.ndarray] = []
    for i, (value, points) in enumerate(zip(raw, candidates)):
        cost = ((points - value) / TICKS_PER_BEAT) ** 2
        if i == 0:
            totals.append(cost)
            links.append(np.full(points.size, -1))
            continue
        previous, prev_total = candidates[i - 1], totals[-1]
        # Running minimum over the previous note's slots, so each slot here
        # can look up the best predecessor strictly before it in O(1).
        best = np.minimum.accumulate(prev_total) if prev_total.size else prev_total
        arg = np.zeros(prev_total.size, dtype=int)
        for j in range(1, prev_total.size):
            arg[j] = j if prev_total[j] < best[j - 1] else arg[j - 1]
        before = np.searchsorted(previous, points, side='left') - 1
        valid = before >= 0
        total = np.full(points.size, np.inf)
        link = np.full(points.size, -1)
        total[valid] = cost[valid] + best[before[valid]]
        link[valid] = arg[before[valid]]
        totals.append(total)
        links.append(link)

    if not totals or not np.isfinite(totals[-1]).any():
        return None
    chosen = np.empty(len(raw))
    j = int(np.argmin(totals[-1]))
    for i in range(len(raw) - 1, -1, -1):
        if j < 0 or not np.isfinite(totals[i][j]):
            return None
        chosen[i] = candidates[i][j]
        j = int(links[i][j])
    return chosen


def assign_ends(onsets: np.ndarray, raw_on: np.ndarray, raw_end: np.ndarray,
                ternary: FrozenSet[int], release: float = 1.0,
                absorb: float = ABSORB_REST_BEATS) -> np.ndarray:
    """Where each note stops, in ticks.

    A gap to the next note shorter than `absorb` beats is articulation, and
    the note is held to the next onset. A longer gap is a rest: the note keeps
    its sounding length, stretched by the track's `release` factor, and stops
    at the nearest grid point after its onset - at least one step long, and
    never past the next note.
    """
    ends = np.empty(len(onsets))
    for i, onset in enumerate(onsets):
        following = onsets[i + 1] if i + 1 < len(onsets) else None
        if following is not None and (
                raw_on[i + 1] - raw_end[i]) / TICKS_PER_BEAT < absorb:
            ends[i] = following
            continue
        target = onset + max(0.0, raw_end[i] - raw_on[i]) * release
        high = following if following is not None else \
            max(target, onset) + 2 * TICKS_PER_BEAT
        points = grid_points(onset + 1, high, ternary)
        # Nearest, and the later of two equally near: releases are early.
        ends[i] = points[int(np.argmin(np.abs(points - target)
                                       - 1e-9 * points))]
    return ends


def _time_at(grid: BeatGrid, position: float) -> float:
    """The inverse of `BeatGrid.beat_positions`: seconds at a beat position,
    continuing at the edge tempo outside the tracked span as it does."""
    beats = grid.beat_times
    period = grid.beat_period
    index = position + grid.offset
    if beats.size < 2:
        return float(index * period)
    if index < 0:
        return float(beats[0] + index * period)
    if index > beats.size - 1:
        return float(beats[-1] + (index - (beats.size - 1)) * period)
    return float(np.interp(index, np.arange(beats.size), beats))


def _release_factor(raw_on: np.ndarray, raw_end: np.ndarray,
                    absorb: float = ABSORB_REST_BEATS) -> float:
    """How much longer a note is written than it sounds, for the whole track.

    Measured where the answer is known: a legato note's written length is
    the gap to the next onset, so the total of those over their total sounding
    length is the singer's release. `rhythm`'s duty estimate was tried first
    and gives up under realistic jitter - it fits durations to a ratio set,
    and a sixteenth +/-30ms matches nothing - after which a quarter note
    released early rounded to a dotted eighth. A ratio of totals rather than
    a median of ratios, because a sixteenth's ratio is mostly jitter and a
    median of them is biased long. Never below 1 (a note is not written
    shorter than it sounds), capped at 1.5, and 1 with fewer than 2 legato
    notes to go on.
    """
    if raw_on.size < 3:
        return 1.0
    sounding = raw_end[:-1] - raw_on[:-1]
    written = raw_on[1:] - raw_on[:-1]
    gap = (raw_on[1:] - raw_end[:-1]) / TICKS_PER_BEAT
    legato = (gap < absorb) & (sounding > 0) & (written > 0)
    if np.count_nonzero(legato) < 2:
        return 1.0
    ratio = float(np.sum(written[legato]) / np.sum(sounding[legato]))
    return float(np.clip(ratio, 1.0, 1.5))


def _times(note) -> Tuple[float, float]:
    start = getattr(note, 'start', None)
    if start is None:
        start = note.onset
    end = getattr(note, 'end', None)
    if end is None:
        end = note.offset
    return float(start), float(end)


def _midi(note) -> float:
    return float(note.midi)


def _name(note) -> str:
    name = getattr(note, 'name', None)
    if isinstance(name, str) and name:
        return name
    from .pitch.engine import midi_to_name
    return midi_to_name(_midi(note))


# --------------------------------------------------------------------------
# Lyrics
# --------------------------------------------------------------------------

def lyric_marks(notes: Sequence, lyrics=None) -> List[Optional[LyricMark]]:
    """For each note (in the order given, which must be time order), the lyric
    to print with it.

    With word timings, a word is printed under the first note it is sung on;
    the notes it is held over get nothing, and the first carries a melisma
    line. With line timings only, a line's text goes on its first note. Notes
    are matched to words and lines by the same rules `lyrics.align` used to
    label them, so the printed word is the note's own `lyric`; the lyric
    objects are needed only to tell a repeated word ('la la la') from one
    held word, which the note labels alone cannot. Without them, a new word
    starts wherever the label changes.
    """
    words = list(getattr(lyrics, 'words', None) or [])
    lines = list(getattr(lyrics, 'lines', None) or [])

    if words:
        kind = 'word'
        ids = _word_ids(notes, words)
        display = _display_words(lyrics)
        text = {k: display.get(id(words[k]), words[k].text) for k in set(ids)
                if k is not None}
    elif lines:
        kind = 'line'
        ids = _line_ids(notes, lines)
        text = {k: lines[k].text for k in set(ids) if k is not None}
    else:
        kind = 'word' if any(getattr(n, 'syllable', None) for n in notes) \
            else 'line'
        ids, text, previous = [], {}, None
        for note in notes:
            label = getattr(note, 'lyric', None)
            if not label:
                ids.append(None)
                previous = None
                continue
            if label != previous or not ids or ids[-1] is None:
                text[len(text)] = label
            ids.append(len(text) - 1)
            previous = label

    marks: List[Optional[LyricMark]] = [None] * len(notes)
    printed = set()
    for i, key in enumerate(ids):
        if key is None or key in printed or not str(text[key]).strip():
            continue
        printed.add(key)
        held = i + 1 < len(ids) and ids[i + 1] == key
        marks[i] = LyricMark(text=str(text[key]).strip(), kind=kind,
                             extend=held and kind == 'word')
    return marks


def _word_ids(notes: Sequence, words: Sequence) -> List[Optional[int]]:
    # The rule of `align._attach_words`: the word a note overlaps most.
    starts = np.array([w.start for w in words], dtype=float)
    ends = np.array([w.end for w in words], dtype=float)
    ids: List[Optional[int]] = []
    for note in notes:
        start, end = _times(note)
        overlap = np.minimum(ends, end) - np.maximum(starts, start)
        best = int(np.argmax(overlap))
        ids.append(best if overlap[best] > 0 else None)
    return ids


def _line_ids(notes: Sequence, lines: Sequence) -> List[Optional[int]]:
    # The rule of `align._attach_lines`: the last line started, if not ended.
    starts = np.array([line.start for line in lines], dtype=float)
    ids: List[Optional[int]] = []
    for note in notes:
        start, _ = _times(note)
        idx = int(np.searchsorted(starts, start, side='right') - 1)
        if idx < 0:
            ids.append(None)
            continue
        line = lines[idx]
        ids.append(idx if line.end is None or start < line.end else None)
    return ids


def _display_words(lyrics) -> Dict[int, str]:
    """Aligned words are lower-cased and stripped of punctuation for the
    aligner; where a line's words still line up one-for-one with its text,
    print the lyric sheet's own spelling instead."""
    display: Dict[int, str] = {}
    for line in getattr(lyrics, 'lines', None) or []:
        written = line.word_texts
        if line.words and len(line.words) == len(written):
            for word, text in zip(line.words, written):
                display[id(word)] = text
    return display


# --------------------------------------------------------------------------
# Layout: measures, note values, ties and triplets
# --------------------------------------------------------------------------

def layout(score: QuantizedScore) -> List[Measure]:
    """Cut the score into measures of printable symbols.

    Notes and rests are split at bar lines (tied), and within a bar into
    values a reader expects: a note that starts off the beat is tied over the
    next beat rather than written as a value that hides it, and a note on a
    beat takes the longest value that may start there. A beat written in
    triplets carries a bracket around its own notes only.
    """
    bar = score.bar_ticks
    end = score.end
    starts = [0] if score.pickup else []
    starts.extend(range(score.pickup, end, bar))
    bounds = list(zip(starts, starts[1:] + [end]))

    measures = [Measure(number=(0 if score.pickup and i == 0
                                else i + (0 if score.pickup else 1)),
                        start=s, length=e - s)
                for i, (s, e) in enumerate(bounds)]

    # Everything in the score as (start, end, note-or-None), gaps as rests.
    events: List[Tuple[int, int, Optional[QuantizedNote]]] = []
    cursor = 0
    for note in score.notes:
        if note.start > cursor:
            events.append((cursor, note.start, None))
        events.append((note.start, note.end, note))
        cursor = note.end
    if cursor < end:
        events.append((cursor, end, None))

    index = 0
    for measure in measures:
        m_start, m_end = measure.start, measure.start + measure.length
        while index < len(events) and events[index][1] <= m_start:
            index += 1
        j = index
        while j < len(events) and events[j][0] < m_end:
            e_start, e_end, note = events[j]
            s, e = max(e_start, m_start), min(e_end, m_end)
            spans = _split(s, e, measure, score.beats_per_bar,
                           rest=note is None)
            for k, (p_start, p_len) in enumerate(spans):
                measure.pieces.append(Piece(
                    start=p_start, duration=p_len, note=note,
                    tie_start=note is not None and (
                        k < len(spans) - 1 or e < e_end),
                    tie_stop=note is not None and (k > 0 or s > e_start)))
            j += 1
        if all(p.is_rest for p in measure.pieces) and not measure.implicit:
            measure.pieces = [Piece(start=m_start, duration=measure.length,
                                    measure_rest=True, type='whole')]
        _name_values(measure, score.ternary_beats)
    return measures


def _split(start: int, end: int, measure: Measure, beats_per_bar: int,
           rest: bool) -> List[Tuple[int, int]]:
    """[start, end) within one measure as (start, length) printable spans."""
    spans: List[Tuple[int, int]] = []
    pos = start
    while pos < end:
        beat_start = (pos // TICKS_PER_BEAT) * TICKS_PER_BEAT
        offset = pos - beat_start
        remaining = end - pos
        if offset:
            # Off the beat: never past the next beat, so the beat stays visible.
            stop = min(end, beat_start + TICKS_PER_BEAT)
            if rest and stop - pos == 9:
                # A dotted rest starting on the 'e' hides the eighth.
                spans.append((pos, 3))
                pos += 3
                continue
            spans.append((pos, stop - pos))
            pos = stop
        elif remaining < TICKS_PER_BEAT:
            spans.append((pos, remaining))
            pos = end
        else:
            length = _on_beat_length((pos - measure.start) // TICKS_PER_BEAT,
                                     remaining, beats_per_bar, rest,
                                     measure.implicit)
            spans.append((pos, length))
            pos += length
    return spans


def _on_beat_length(beat: int, remaining: int, beats_per_bar: int,
                    rest: bool, pickup: bool) -> int:
    """The longest single value that may start on this beat of the bar.

    Conventional placements, not a dogma: a whole note only on the downbeat,
    a dotted half on beat 1 or 2, a half on beats 1-3 in 4/4 (the syncopated
    half on 2 reads easily in a pop melody) but a half rest only on 1 or 3,
    and a dotted quarter only when it completes the note - otherwise the tied
    quarter shows where the next beat falls. In a pickup, whose beats do not
    line up with the bar, only whole beats are used.
    """
    if pickup:
        return TICKS_PER_BEAT
    options = []
    if beats_per_bar == 4 and beat == 0:
        options.append(_WHOLE)
    if beat == 0 or (beats_per_bar == 4 and beat == 1 and not rest):
        options.append(_DOTTED_HALF)
    half_beats = {0, 1, 2} if not rest else {0, 2}
    if beats_per_bar == 3:
        half_beats = {0, 1}
    if beats_per_bar == 2:
        half_beats = {0}
    if beat in half_beats:
        options.append(_HALF)
    if not rest:
        options.append(_DOTTED_QUARTER)
    for length in options:
        if length == _DOTTED_QUARTER and remaining != length:
            continue
        if length <= remaining:
            return length
    return TICKS_PER_BEAT


def _name_values(measure: Measure, ternary_beats: FrozenSet[int]) -> None:
    """Fill in each piece's type, dots and triplet marks."""
    groups: Dict[int, List[Piece]] = {}
    for piece in measure.pieces:
        if piece.measure_rest:
            continue
        beat = piece.start // TICKS_PER_BEAT
        if beat in ternary_beats and piece.duration < TICKS_PER_BEAT:
            piece.triplet = True
            piece.type, piece.dots = _TRIPLET_VALUES[piece.duration]
            groups.setdefault(beat, []).append(piece)
        else:
            piece.type, piece.dots = _VALUES[piece.duration]
    for pieces in groups.values():
        pieces[0].tuplet_start = True
        pieces[-1].tuplet_stop = True


def check_layout(measures: Sequence[Measure]) -> None:
    """Raise if any measure's symbols do not add up to exactly its length -
    the invariant every notation program depends on."""
    for measure in measures:
        total = sum(p.duration for p in measure.pieces)
        if total != measure.length:
            raise AssertionError(
                f"measure {measure.number}: {total} ticks, expected "
                f"{measure.length}")
        cursor = measure.start
        for piece in measure.pieces:
            if piece.start != cursor:
                raise AssertionError(
                    f"measure {measure.number}: gap or overlap at {cursor}")
            cursor += piece.duration
