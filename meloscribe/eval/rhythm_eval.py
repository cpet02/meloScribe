"""Does the rhythmic plausibility score actually separate good from bad?

A plausibility metric that does not go down when the transcription gets worse
is decoration, so this harness builds that comparison directly rather than
inferring it. For each metrical benchmark case it detects the beat grid from
the rendered audio - the real path, not a handed-in tempo - and then scores the
ground-truth notes alongside deliberately corrupted versions of the same notes.

The corruptions are the failure modes the pitch engine actually has, not
abstract noise:

    tight       timing error a good transcription really makes (+/-25ms)
    loose       onsets smeared by a quarter beat
    shred       held notes split into fragments - the vibrato-splitting bug
                that cost note F1 0.58 before the attack gate was added
    merge       adjacent notes run together - the opposite failure
    scramble    onsets and durations drawn at random: the floor

`scramble` is the important row. Because the score is chance-normalised it
should sit near 0.0, and if it does not, the normalisation is wrong and every
other number here is inflated.

    python -m meloscribe.eval.rhythm_eval
    python -m meloscribe.eval.rhythm_eval --cases metrical_swing --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .. import rhythm as rhythm_mod
from ..rhythm import BeatGrid, RhythmReport
from . import synth
from .groundtruth import Note

DEFAULT_AUDIO_DIR = Path('data/eval/synthetic')


# --------------------------------------------------------------------------
# Corruptions
# --------------------------------------------------------------------------

def _shift(notes: Sequence[Note], deltas: np.ndarray) -> List[Note]:
    """Move onsets while keeping each note's length - so only timing changes."""
    return [Note(onset=max(0.0, n.onset + float(d)),
                 offset=max(0.0, n.onset + float(d)) + n.duration,
                 midi=n.midi)
            for n, d in zip(notes, deltas)]


def corrupt_tight(notes: Sequence[Note], period: float,
                  rng: np.random.Generator) -> List[Note]:
    """The timing error a *correct* transcription still has.

    10ms frames plus a soft vocal attack put real onsets a couple of frames
    from truth. This row is the control: if it scores much below clean, the
    metric is too strict to use on anything real.
    """
    return _shift(notes, rng.normal(0.0, 0.025, len(notes)))


def corrupt_loose(notes: Sequence[Note], period: float,
                  rng: np.random.Generator) -> List[Note]:
    """Onsets smeared by up to a quarter of a beat in either direction."""
    return _shift(notes, rng.uniform(-0.25, 0.25, len(notes)) * period)


def corrupt_shred(notes: Sequence[Note], period: float,
                  rng: np.random.Generator) -> List[Note]:
    """Split every sustained note into three fragments.

    This is a real regression, not an invented one: before the amplitude-attack
    gate, vibrato peaks in the onset activation split held notes into ~200ms
    pieces at the vibrato rate. The fragments land nowhere near the grid and
    have durations unrelated to any note value, so both components of the score
    should react.
    """
    out: List[Note] = []
    for note in notes:
        if note.duration < 0.4 * period:
            out.append(note)
            continue
        pieces = 3
        length = note.duration / pieces
        for k in range(pieces):
            start = note.onset + k * length
            out.append(Note(onset=start, offset=start + length * 0.9,
                            midi=note.midi))
    return out


def corrupt_merge(notes: Sequence[Note], period: float,
                  rng: np.random.Generator) -> List[Note]:
    """Run adjacent notes together, halving the note count.

    The mirror image of shredding: durations become sums of grid values, which
    are often themselves grid values, so this should hurt the duration
    component far less than the onset component. If it hurts neither, the score
    is blind to under-segmentation.
    """
    out: List[Note] = []
    for i in range(0, len(notes) - 1, 2):
        first, second = notes[i], notes[i + 1]
        out.append(Note(onset=first.onset, offset=second.offset,
                        midi=first.midi))
    if len(notes) % 2:
        out.append(notes[-1])
    return out


def corrupt_scramble(notes: Sequence[Note], period: float,
                     rng: np.random.Generator) -> List[Note]:
    """Random onsets and durations over the same span: the floor."""
    span_start = min(n.onset for n in notes)
    span_end = max(n.offset for n in notes)
    onsets = np.sort(rng.uniform(span_start, span_end, len(notes)))
    lengths = np.exp(rng.uniform(np.log(0.12 * period),
                                 np.log(2.0 * period), len(notes)))
    return [Note(onset=float(o), offset=float(o + d), midi=n.midi)
            for o, d, n in zip(onsets, lengths, notes)]


Corruption = Callable[[Sequence[Note], float, np.random.Generator], List[Note]]

CORRUPTIONS: Dict[str, Optional[Corruption]] = {
    'clean': None,
    'tight': corrupt_tight,
    'loose': corrupt_loose,
    'shred': corrupt_shred,
    'merge': corrupt_merge,
    'scramble': corrupt_scramble,
}


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def tempo_relation(detected: float, true_bpm: float,
                   tolerance: float = 0.04) -> str:
    """Describe a detected tempo against the truth, octave errors named.

    Reported rather than silently forgiven: a doubled tempo is a real defect
    even when the plausibility score survives it, and the whole point of the
    diagnostics is to notice it.
    """
    if detected <= 0 or true_bpm <= 0:
        return 'none'
    for factor, label in ((1.0, 'ok'), (2.0, 'double'), (0.5, 'half'),
                          (3.0, 'triple'), (1 / 3, 'third'),
                          (4.0, 'quadruple'), (0.25, 'quarter')):
        if abs(detected / (true_bpm * factor) - 1.0) <= tolerance:
            return label
    return 'wrong'


def evaluate_case(case: Dict, audio_dir: Path, seed: int = 0,
                  force_audio: bool = False) -> Dict:
    """Detect the grid for one case and score every corruption against it."""
    truth = synth.build_dataset(audio_dir, cases=[case], force=force_audio)[0]
    notes = list(truth.notes or [])

    grid = rhythm_mod.detect_grid(truth.audio_path)
    period = grid.beat_period or 60.0 / case['bpm']
    rng = np.random.default_rng(seed)

    scores: Dict[str, RhythmReport] = {}
    for name, corruption in CORRUPTIONS.items():
        variant = list(notes) if corruption is None else corruption(
            notes, period, rng)
        # A fresh grid per variant: `analyse` fits the global phase offset, and
        # sharing one grid would let a clean run's offset flatter a corrupted
        # one (or the reverse).
        variant_grid = BeatGrid(bpm=grid.bpm, beat_times=grid.beat_times,
                                subdivisions=grid.subdivisions,
                                confidence=grid.confidence,
                                warnings=list(grid.warnings))
        scores[name] = rhythm_mod.analyse(variant, variant_grid)

    return {'case': case['name'],
            'true_bpm': case['bpm'],
            'detected_bpm': grid.bpm,
            'tempo': tempo_relation(grid.bpm, case['bpm']),
            'tempo_confidence': grid.confidence,
            'n_notes': len(notes),
            'reports': scores,
            'octaves': _score_tempo_octaves(notes, grid),
            'grid_warnings': grid.warnings}


def _score_tempo_octaves(notes: Sequence[Note],
                         grid: BeatGrid) -> Dict[str, float]:
    """Score correct notes against deliberately wrong metrical levels.

    The named pitfall: a beat tracker that reports double or half tempo makes a
    correct rhythm look implausible, or makes everything look aligned. Relying
    on the tracker to actually make that mistake is not a test - on this
    material it stubbornly gets the tempo right, including on the half-time
    case built to fool it - so the wrong grids are constructed directly.
    """
    variants = {'true': grid.beat_times,
                # Half tempo: keep every other beat.
                'half': grid.beat_times[::2],
                # Double tempo: insert a beat at every midpoint.
                'double': np.sort(np.concatenate([
                    grid.beat_times,
                    grid.beat_times[:-1] + np.diff(grid.beat_times) / 2]))}

    out: Dict[str, float] = {}
    for name, beats in variants.items():
        if beats.size < 4:
            out[name] = float('nan')
            continue
        period = float(np.median(np.diff(beats)))
        wrong = BeatGrid(bpm=60.0 / period if period > 0 else 0.0,
                         beat_times=beats, subdivisions=grid.subdivisions,
                         confidence=grid.confidence)
        out[name] = rhythm_mod.analyse(list(notes), wrong).plausibility
    return out


def run(cases: Optional[List[Dict]] = None, audio_dir=DEFAULT_AUDIO_DIR,
        seed: int = 0, force_audio: bool = False) -> List[Dict]:
    cases = cases or synth.METRICAL_CASES
    audio_dir = Path(audio_dir)
    return [evaluate_case(case, audio_dir, seed=seed, force_audio=force_audio)
            for case in cases]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def format_report(rows: List[Dict], verbose: bool = False) -> str:
    if not rows:
        return '(no cases)'

    names = list(CORRUPTIONS)
    out: List[str] = []

    header = ['case', 'bpm', 'detected', 'tempo', 'conf'] + names
    table = [header]
    for row in rows:
        table.append([row['case'][:22], f"{row['true_bpm']:.0f}",
                      f"{row['detected_bpm']:.1f}", row['tempo'],
                      f"{row['tempo_confidence']:.2f}"]
                     + [f"{row['reports'][n].plausibility:.3f}" for n in names])

    means = ['MEAN', '', '', '',
             f"{np.mean([r['tempo_confidence'] for r in rows]):.2f}"]
    means += [f"{np.mean([r['reports'][n].plausibility for r in rows]):.3f}"
              for n in names]
    table.append(means)

    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    line = '  '.join(h.ljust(widths[i]) for i, h in enumerate(header))
    out += ['plausibility by corruption', line, '-' * len(line)]
    out += ['  '.join(c.ljust(widths[i]) for i, c in enumerate(r))
            for r in table[1:-1]]
    out += ['-' * len(line),
            '  '.join(c.ljust(widths[i]) for i, c in enumerate(table[-1]))]

    clean = np.mean([r['reports']['clean'].plausibility for r in rows])
    scramble = np.mean([r['reports']['scramble'].plausibility for r in rows])
    tight = np.mean([r['reports']['tight'].plausibility for r in rows])
    worst_bad = max(np.mean([r['reports'][n].plausibility for r in rows])
                    for n in ('loose', 'shred', 'scramble'))

    out += ['', f"separation: clean {clean:.3f} vs worst corruption "
                f"{worst_bad:.3f}  (gap {clean - worst_bad:+.3f})",
            f"floor:      scramble scores {scramble:.3f} "
            f"(chance-normalised, so this should be near 0)",
            f"tolerance:  realistic timing error costs "
            f"{clean - tight:+.3f}"]

    octaves = {k: np.nanmean([r['octaves'][k] for r in rows])
               for k in ('true', 'half', 'double')}
    out += ['', f"metrical level: correct notes score {octaves['true']:.3f} on "
                f"the tracked grid, {octaves['half']:.3f} at half tempo, "
                f"{octaves['double']:.3f} at double"]

    bad_tempo = [r['case'] for r in rows if r['tempo'] != 'ok']
    if bad_tempo:
        out.append(f"tempo:      wrong metrical level on {', '.join(bad_tempo)}")

    unreliable = [r['case'] for r in rows if not r['reports']['clean'].reliable]
    if unreliable:
        out.append(f"unreliable: {', '.join(unreliable)} "
                   f"(grid not trusted; scores above are informational)")

    if verbose:
        out.append('')
        for row in rows:
            report = row['reports']['clean']
            out.append(f"{row['case']}: onset {report.onset_score:.3f}  "
                       f"duration {report.duration_score:.3f}  "
                       f"on-grid {report.on_grid_fraction:.0%}  "
                       f"offset {report.grid.offset:+.3f} beat")
            for warning in row['grid_warnings']:
                out.append(f"    ! {warning}")

    return '\n'.join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='meloscribe.eval.rhythm_eval',
        description='Check that rhythmic plausibility separates good '
                    'transcriptions from bad ones.')
    parser.add_argument('--cases', help='Comma-separated case names')
    parser.add_argument('--audio-dir', default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--force-audio', action='store_true',
                        help='Re-render the WAVs even if they exist')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args(argv)

    cases = synth.METRICAL_CASES
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(',')}
        cases = [c for c in cases if c['name'] in wanted]
        if not cases:
            print(f"error: no cases match {sorted(wanted)}", file=sys.stderr)
            return 1

    rows = run(cases, audio_dir=args.audio_dir, seed=args.seed,
               force_audio=args.force_audio)
    print(format_report(rows, verbose=args.verbose))
    return 0


if __name__ == '__main__':
    sys.exit(main())
