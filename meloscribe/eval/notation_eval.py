"""Does quantisation write what a musician would have written?

Sheet music is only useful if a note comes out as the value on the page: a
quarter note sung 92% long must still read as a quarter, an off-beat eighth as
an off-beat eighth. This harness scores exactly that, on melodies whose ground
truth is *written in beats*:

    onset   the quantised note starts on its written beat position
    value   it has its written length (rests shorter than a written rest are
            articulation, and must be absorbed)
    both    both at once - the note reads exactly as written

Three sources of error are kept apart by scoring each case against:

    true      the tempo and phase the melody was written at: the quantiser
              alone, with a perfect grid
    tracked   what `rhythm.detect_grid` finds in the rendered audio: the grid
              the export actually uses
    fallback  no grid at all - the fixed tempo from note spacing that stands
              in when the tracker is not trusted (scored at whichever of
              half, same or double speed fits, since a guessed tempo is free
              to land on another metrical level)

each at no timing error and at Gaussian jitter on every onset and release.
`--transcribe` adds the real thing: the pitch engine's own notes from the
rendered audio, matched to the written ones.

Swing is scored but kept out of the mean. Swung eighths are written straight
and played long-short; which of those a transcription should show is a
question of style, not of accuracy, and any single answer would be scored
wrong on half the music.

    python -m meloscribe.eval.notation_eval
    python -m meloscribe.eval.notation_eval --transcribe --verbose
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import notation, rhythm
from . import synth
from .groundtruth import Note

DEFAULT_AUDIO_DIR = Path('data/eval/synthetic')
TICKS = notation.TICKS_PER_BEAT

# Melodies beyond `synth`'s metrical set, for the things it does not contain:
# a pickup, notes tied over beats and bar lines, dotted figures, triplets
# mixed into straight time, and long values. (midi or None for a rest,
# length in beats.)
EXTRA_PATTERNS: Dict[str, List[Tuple[Optional[int], float]]] = {
    'pickup': [(67, 0.5), (69, 0.5), (71, 1.5), (69, 0.5), (67, 1.0),
               (64, 1.0), (62, 2.0), (None, 1.0), (62, 0.5), (64, 0.5),
               (67, 3.0), (None, 1.0)],
    'ties': [(60, 1.5), (62, 1.5), (64, 1.0), (65, 0.5), (67, 2.5),
             (69, 1.0), (67, 3.0), (None, 1.0), (64, 0.75), (62, 0.75),
             (60, 2.5)],
    'dotted': [(60, 0.75), (62, 0.25), (64, 0.75), (65, 0.25), (67, 0.25),
               (69, 0.5), (67, 0.25), (65, 1.0), (64, 0.75), (62, 0.25),
               (60, 2.0), (None, 1.0)],
    'mixed_triplets': [(67, 0.5), (69, 0.5), (71, 1 / 3), (72, 1 / 3),
                       (71, 1 / 3), (69, 1.0), (67, 0.5), (65, 0.5),
                       (64, 2 / 3), (62, 1 / 3), (60, 2.0), (None, 1.0)],
    'long': [(60, 4.0), (None, 4.0), (64, 2.0), (67, 2.0), (72, 6.0),
             (None, 2.0)],
}

EXTRA_CASES: List[Dict] = [
    {'name': 'notation_pickup', 'pattern': 'pickup', 'bpm': 76.0},
    {'name': 'notation_ties', 'pattern': 'ties', 'bpm': 104.0},
    {'name': 'notation_dotted', 'pattern': 'dotted', 'bpm': 88.0},
    {'name': 'notation_triplets', 'pattern': 'mixed_triplets', 'bpm': 112.0},
    {'name': 'notation_long', 'pattern': 'long', 'bpm': 126.0},
]

# The count-in `synth.build_metrical_notes` puts before every melody.
COUNT_IN_BEATS = 2.0
LEGATO = 0.92


@dataclass
class Written:
    """One note as written: beat position and value, both in ticks."""
    onset: int
    value: int
    midi: float


@dataclass
class Case:
    name: str
    bpm: float
    written: List[Written]
    performed: List[Note]
    audio: Optional[Path] = None
    swing: bool = False


@dataclass
class Score:
    """How one quantisation compared with what was written."""
    onset: float
    value: float
    both: float
    matched: float = 1.0          # fraction of written notes with a partner
    level: float = 1.0            # metrical level the grid landed on
    triplet_beats: Tuple[int, int, int] = (0, 0, 0)  # found, true, correct
    detail: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def _written(pattern: Sequence[Tuple[Optional[int], float]]) -> List[Written]:
    out, beat = [], COUNT_IN_BEATS
    for midi, length in pattern:
        if midi is not None:
            out.append(Written(onset=int(round(beat * TICKS)),
                               value=int(round(length * TICKS)),
                               midi=float(midi)))
        beat += length
    return out


def _lay(pattern, bpm: float) -> List[Note]:
    """A pattern performed as `synth.build_metrical_notes` performs one:
    after the same count-in, each note released at the same legato."""
    period = 60.0 / bpm
    out, beat = [], COUNT_IN_BEATS
    for midi, length in pattern:
        if midi is not None:
            onset = beat * period
            out.append(Note(onset=onset, offset=onset + length * period * LEGATO,
                            midi=float(midi)))
        beat += length
    return out


def build_cases(audio_dir: Path, force: bool = False) -> List[Case]:
    """The metrical benchmark plus the extra patterns, rendered to audio so
    the real beat tracker has something to track."""
    audio_dir = Path(audio_dir)
    cases: List[Case] = []
    for spec in synth.METRICAL_CASES:
        truth = synth.build_dataset(audio_dir, cases=[spec], force=force)[0]
        cases.append(Case(name=spec['name'], bpm=spec['bpm'],
                          written=_written(synth._METRICAL[spec['pattern']]),
                          performed=list(truth.notes or []),
                          audio=truth.audio_path,
                          swing=bool(spec.get('swing'))))
    for spec in EXTRA_CASES:
        pattern = EXTRA_PATTERNS[spec['pattern']]
        notes = _lay(pattern, spec['bpm'])
        path = audio_dir / f"{spec['name']}.wav"
        if force or not path.exists():
            _render(notes, spec, path)
        cases.append(Case(name=spec['name'], bpm=spec['bpm'],
                          written=_written(pattern), performed=notes,
                          audio=path))
    return cases


def _render(notes: List[Note], spec: Dict, path: Path) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    audio, _ = synth.render_melody(notes, synth.PRESETS['vocal'],
                                   synth.SAMPLE_RATE,
                                   seed=synth.case_seed(spec['name']),
                                   pulse_bpm=spec['bpm'], pulse_level=0.35)
    sf.write(str(path), audio, synth.SAMPLE_RATE)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def jitter(notes: Sequence[Note], sigma: float,
           rng: np.random.Generator) -> List[Note]:
    """Independent Gaussian error on every onset and every release, as a
    transcription has: its onsets and its note-offs are separate decisions."""
    out = []
    for note in notes:
        onset = max(0.0, note.onset + float(rng.normal(0.0, sigma)))
        offset = note.offset + float(rng.normal(0.0, sigma))
        out.append(Note(onset=onset, offset=max(onset + 0.03, offset),
                        midi=note.midi))
    return out


def score_quantized(written: Sequence[Written],
                    placed: Sequence[Optional[notation.QuantizedNote]],
                    ternary: Sequence[int] = (),
                    levels: Sequence[float] = (1.0,)) -> Score:
    """Compare quantised notes with the written ones they stand for.

    `placed[i]` is the quantised partner of `written[i]`, or None. Positions
    are compared after the one shift in whole beats that lines the most
    notes up - where bar 1 falls is not what is being scored here - and, for
    a guessed tempo, at the metrical level (`levels`) that fits best.
    """
    best: Optional[Score] = None
    for level in levels:
        pairs = [(w, q) for w, q in zip(written, placed) if q is not None]
        shifts: Dict[int, int] = {}
        for w, q in pairs:
            shift = q.start / level - w.onset
            if abs(shift - round(shift)) < 1e-6 and round(shift) % TICKS == 0:
                shifts[int(round(shift))] = shifts.get(int(round(shift)), 0) + 1
        shift = max(shifts, key=shifts.get) if shifts else 0
        onset = [q is not None and abs(q.start / level - shift - w.onset) < 1e-6
                 for w, q in zip(written, placed)]
        value = [q is not None and abs(q.duration / level - w.value) < 1e-6
                 for w, q in zip(written, placed)]
        both = [a and b for a, b in zip(onset, value)]
        result = Score(onset=float(np.mean(onset)), value=float(np.mean(value)),
                       both=float(np.mean(both)),
                       matched=len(pairs) / max(1, len(written)), level=level)
        result.triplet_beats = _triplet_agreement(written, ternary,
                                                  shift / TICKS, level)
        result.detail = [
            f"{'ok ' if b else 'BAD'} written {w.onset / TICKS:6.3f}+"
            f"{w.value / TICKS:.3f}  got "
            + (f"{q.start / level / TICKS - shift / TICKS:6.3f}+"
               f"{q.duration / level / TICKS:.3f}" if q else 'nothing')
            for w, q, b in zip(written, placed, both)]
        if best is None or result.both > best.both:
            best = result
    return best


def _triplet_agreement(written: Sequence[Written], ternary: Sequence[int],
                       shift_beats: float, level: float) -> Tuple[int, int, int]:
    """(beats written as triplets by us, beats that are triplets, overlap)."""
    truth = {w.onset // TICKS for w in written if w.onset % 4 == 0
             and w.onset % 3 != 0}
    truth |= {(w.onset + w.value) // TICKS for w in written
              if (w.onset + w.value) % 4 == 0 and (w.onset + w.value) % 3 != 0}
    found = {int(round(b / level - shift_beats)) for b in ternary} \
        if level == 1.0 else set()
    return len(found), len(truth), len(found & truth)


def quantize_case(case: Case, notes: Sequence[Note], grid_kind: str,
                  tracked: Optional[rhythm.BeatGrid]) -> notation.QuantizedScore:
    if grid_kind == 'true':
        span = max(n.offset for n in notes) + 4.0
        grid = rhythm.constant_tempo_grid(case.bpm, span, start=0.0)
    elif grid_kind == 'tracked':
        grid = tracked
    else:
        grid = None
    return notation.quantize(notes, grid)


def _absolute(score: notation.QuantizedScore) -> List[notation.QuantizedNote]:
    """The score's notes in input order."""
    out: List[Optional[notation.QuantizedNote]] = [None] * len(score.notes)
    for note in score.notes:
        out[note.source] = note
    return out


def evaluate_case(case: Case, jitters: Sequence[float], seeds: int,
                  transcribe: bool = False) -> Dict:
    tracked = rhythm.detect_grid(case.audio) if case.audio else None
    rows: Dict[Tuple[str, float], List[Score]] = {}
    for grid_kind in ('true', 'tracked', 'fallback'):
        levels = (0.5, 1.0, 2.0) if grid_kind == 'fallback' else (1.0,)
        for sigma in jitters:
            runs = 1 if sigma == 0 else seeds
            for seed in range(runs):
                rng = np.random.default_rng(1000 * seed + 17)
                notes = case.performed if sigma == 0 else jitter(
                    case.performed, sigma, rng)
                score = quantize_case(case, notes, grid_kind, tracked)
                rows.setdefault((grid_kind, sigma), []).append(
                    score_quantized(case.written, _absolute(score),
                                    sorted(score.ternary_beats), levels))
    result = {'case': case.name, 'bpm': case.bpm, 'swing': case.swing,
              'n': len(case.written), 'rows': rows,
              'tracked_bpm': tracked.bpm if tracked else 0.0,
              'tracked_reliable': bool(tracked and tracked.reliable)}
    if transcribe and case.audio:
        result['transcribed'] = _transcribed(case, tracked)
    return result


def _transcribed(case: Case, tracked: Optional[rhythm.BeatGrid]) -> Score:
    """The pitch engine's notes from the rendered audio, quantised on the
    tracked grid and matched one-to-one to the written notes."""
    from .systems import get_system

    prediction = get_system('ensemble').transcribe(case.audio)
    notes = list(prediction.notes or [])
    if not notes:
        return Score(onset=0.0, value=0.0, both=0.0, matched=0.0)
    score = notation.quantize(notes, tracked)
    placed = _absolute(score)
    # Pair each written note with the nearest unused transcribed onset within
    # a quarter of a beat - no further, or a missed note would borrow its
    # neighbour's.
    period = 60.0 / case.bpm
    used = set()
    partners: List[Optional[notation.QuantizedNote]] = []
    for written, performed in zip(case.written, case.performed):
        best, distance = None, 0.25 * period
        for i, note in enumerate(notes):
            gap = abs(note.onset - performed.onset)
            if i not in used and gap <= distance and \
                    abs(note.midi - written.midi) < 0.5:
                best, distance = i, gap
        if best is not None:
            used.add(best)
        partners.append(placed[best] if best is not None else None)
    result = score_quantized(case.written, partners,
                             sorted(score.ternary_beats))
    result.detail.append(f"transcribed {len(notes)} notes for "
                         f"{len(case.written)} written")
    return result


def run(audio_dir=DEFAULT_AUDIO_DIR, jitters: Sequence[float] = (0.0, 0.025),
        seeds: int = 10, transcribe: bool = False,
        names: Optional[Sequence[str]] = None) -> List[Dict]:
    cases = build_cases(Path(audio_dir))
    if names:
        cases = [c for c in cases if c.name in set(names)]
    return [evaluate_case(c, jitters, seeds, transcribe) for c in cases]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _mean(scores: Sequence[Score], attr: str) -> float:
    return float(np.mean([getattr(s, attr) for s in scores]))


def format_report(rows: List[Dict], verbose: bool = False) -> str:
    if not rows:
        return '(no cases)'
    keys = sorted({k for r in rows for k in r['rows']},
                  key=lambda k: (('true', 'tracked', 'fallback').index(k[0]),
                                 k[1]))
    out = ['fraction of written notes read back exactly: onset / value / both',
           '']
    header = ['case', 'bpm', 'n'] + [
        f"{grid}{'' if sigma == 0 else f'+{sigma * 1000:.0f}ms'}"
        for grid, sigma in keys]
    table = [header]
    for row in rows:
        cells = [row['case'][:20], f"{row['bpm']:.0f}", str(row['n'])]
        for key in keys:
            scores = row['rows'][key]
            cells.append(f"{_mean(scores, 'onset'):.2f}/"
                         f"{_mean(scores, 'value'):.2f}/"
                         f"{_mean(scores, 'both'):.2f}")
        table.append(cells)

    counted = [r for r in rows if not r['swing']]
    for label, subset, weight in (('MEAN (no swing)', counted, False),
                                  ('NOTES (no swing)', counted, True)):
        cells = [label, '', str(sum(r['n'] for r in subset))]
        for key in keys:
            values = []
            for attr in ('onset', 'value', 'both'):
                per = [_mean(r['rows'][key], attr) for r in subset]
                w = [r['n'] for r in subset] if weight else None
                values.append(float(np.average(per, weights=w)))
            cells.append('/'.join(f"{v:.2f}" for v in values))
        table.append(cells)

    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    line = '  '.join(h.ljust(widths[i]) for i, h in enumerate(header))
    out += [line, '-' * len(line)]
    out += ['  '.join(c.ljust(widths[i]) for i, c in enumerate(r))
            for r in table[1:-2]]
    out += ['-' * len(line)]
    out += ['  '.join(c.ljust(widths[i]) for i, c in enumerate(r))
            for r in table[-2:]]

    # Triplet decisions: does the quantiser invent triplets, or miss them?
    out.append('')
    for key in keys:
        if key[0] == 'fallback':
            continue
        found = true = correct = 0
        for row in counted:
            for s in row['rows'][key]:
                f, t, c = s.triplet_beats
                found, true, correct = found + f, true + t, correct + c
        out.append(f"triplet beats, {key[0]} grid"
                   f"{'' if key[1] == 0 else f' +{key[1] * 1000:.0f}ms'}: "
                   f"{correct} of {true} found, {found - correct} invented")

    levels = [s.level for r in rows for k, v in r['rows'].items()
              if k[0] == 'fallback' for s in v]
    if levels:
        out.append(f"fallback tempo landed at the written level in "
                   f"{np.mean(np.isclose(levels, 1.0)):.0%} of runs")
    unreliable = [r['case'] for r in rows if not r['tracked_reliable']]
    if unreliable:
        out.append(f"tracked grid not trusted (so 'tracked' = fallback): "
                   f"{', '.join(unreliable)}")

    transcribed = [r for r in rows if 'transcribed' in r]
    if transcribed:
        out += ['', 'engine transcription, tracked grid: '
                    'matched / onset / value / both']
        for row in transcribed:
            s = row['transcribed']
            out.append(f"  {row['case']:20s} {s.matched:.2f} / {s.onset:.2f} "
                       f"/ {s.value:.2f} / {s.both:.2f}"
                       + ('   (swing: not in the mean)' if row['swing'] else ''))
        kept = [r['transcribed'] for r in transcribed if not r['swing']]
        weights = [r['n'] for r in transcribed if not r['swing']]
        out.append('  ' + 'NOTES (no swing)'.ljust(20) + ' ' + ' / '.join(
            f"{np.average([getattr(s, a) for s in kept], weights=weights):.2f}"
            for a in ('matched', 'onset', 'value', 'both')))

    if verbose:
        for row in rows:
            clean = row['rows'][('tracked', 0.0)][0]
            out += ['', f"{row['case']} (tracked grid, clean; tracked "
                        f"{row['tracked_bpm']:.1f} BPM)"]
            out += ['    ' + d for d in clean.detail]
            if 'transcribed' in row:
                out += [f"{row['case']} (engine transcription)"]
                out += ['    ' + d for d in row['transcribed'].detail]
    return '\n'.join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='meloscribe.eval.notation_eval',
        description='Score quantisation against melodies written in beats.')
    parser.add_argument('--audio-dir', default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument('--jitter', default='0,25',
                        help='Comma-separated timing error levels, in ms')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--cases', help='Comma-separated case names')
    parser.add_argument('--transcribe', action='store_true',
                        help="Also quantise the pitch engine's own notes")
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args(argv)

    jitters = [float(j) / 1000.0 for j in args.jitter.split(',') if j.strip()]
    names = [c.strip() for c in args.cases.split(',')] if args.cases else None
    rows = run(args.audio_dir, jitters=jitters, seeds=args.seeds,
               transcribe=args.transcribe, names=names)
    print(format_report(rows, verbose=args.verbose))
    return 0


if __name__ == '__main__':
    sys.exit(main())
