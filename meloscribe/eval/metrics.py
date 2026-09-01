"""mir_eval-backed scoring for melody transcription.

Two families of metric, because they fail in different ways and a change that
helps one can quietly hurt the other:

Frame level (mir_eval.melody)
    Voicing Recall / False Alarm  - did we decide "singing" in the right places
    Raw Pitch Accuracy            - correct pitch within 50 cents
    Raw Chroma Accuracy           - correct pitch class, octave errors forgiven
    Overall Accuracy              - the headline single number

Note level (mir_eval.transcription)
    Precision / Recall / F1 with onset and pitch tolerance - whether the
    segmentation produced the right discrete notes, which is what actually
    lands on the page.

`octave_error_rate` is our own diagnostic: the share of voiced frames we got
right modulo the octave but wrong absolutely. It isolates the single failure
mode the harmonic-template voter exists to fix, so it must be tracked
separately rather than buried inside Overall Accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .groundtruth import (DEFAULT_HOP, GroundTruth, Prediction, f0_to_notes,
                          hz_to_midi, resample_f0)

# mir_eval's default: a pitch is "correct" within a quarter tone.
PITCH_TOLERANCE_CENTS = 50.0
# Standard note-transcription tolerances.
ONSET_TOLERANCE_S = 0.05
NOTE_PITCH_TOLERANCE_CENTS = 50.0


@dataclass
class ScoreResult:
    """Scores for one track under one system."""
    track: str
    system: str
    runtime_s: float = 0.0
    frame: Dict[str, float] = field(default_factory=dict)
    note: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def flat(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {'track': self.track, 'system': self.system,
                               'runtime_s': round(self.runtime_s, 2)}
        row.update({k: round(v, 4) for k, v in self.frame.items()})
        row.update({f"note_{k}": round(v, 4) for k, v in self.note.items()})
        return row

    @property
    def headline(self) -> float:
        """Overall Accuracy - the one number to watch across a whole run."""
        return self.frame.get('Overall Accuracy', 0.0)


def _frame_scores(ref: GroundTruth, est: Prediction,
                  hop: float = DEFAULT_HOP) -> Dict[str, float]:
    import mir_eval

    # Score on the reference's own grid when it has one, so we never invent
    # reference frames that the annotator did not actually label.
    if ref.times.size:
        grid = ref.times
        ref_freqs = ref.freqs
    else:
        duration = float(est.times[-1]) if est.times.size else 0.0
        grid = np.arange(int(np.ceil(duration / hop))) * hop
        ref_freqs = np.zeros_like(grid)

    est_freqs = resample_f0(est.times, est.freqs, grid)

    ref_voicing = ref_freqs > 0
    est_voicing = est_freqs > 0

    scores = mir_eval.melody.evaluate(
        grid, ref_freqs, grid, est_freqs,
        cent_tolerance=PITCH_TOLERANCE_CENTS,
    )
    out = {k: float(v) for k, v in scores.items()}

    both_voiced = ref_voicing & est_voicing
    if np.any(both_voiced):
        ref_midi = hz_to_midi(ref_freqs[both_voiced])
        est_midi = hz_to_midi(est_freqs[both_voiced])
        diff = est_midi - ref_midi
        tol = PITCH_TOLERANCE_CENTS / 100.0
        # Right pitch class, wrong octave: the error the template voter targets.
        chroma_ok = np.abs(diff - 12.0 * np.round(diff / 12.0)) <= tol
        exact_ok = np.abs(diff) <= tol
        out['octave_error_rate'] = float(np.mean(chroma_ok & ~exact_ok))
    else:
        out['octave_error_rate'] = 0.0

    return out


def _note_scores(ref: GroundTruth, est: Prediction) -> Dict[str, float]:
    import mir_eval

    ref_notes = ref.notes if ref.notes else f0_to_notes(ref.times, ref.freqs)
    est_notes = est.notes if est.notes else f0_to_notes(est.times, est.freqs)

    if not ref_notes or not est_notes:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'n_ref': float(len(ref_notes)), 'n_est': float(len(est_notes))}

    def unpack(notes):
        intervals = np.array([[n.onset, n.offset] for n in notes], dtype=float)
        pitches = np.array([440.0 * 2 ** ((n.midi - 69) / 12) for n in notes],
                           dtype=float)
        return intervals, pitches

    ref_iv, ref_hz = unpack(ref_notes)
    est_iv, est_hz = unpack(est_notes)

    # offset_ratio=None scores onset+pitch only. Note-off timing in sung melody
    # is genuinely ambiguous, so grading it would add noise, not signal.
    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, est_iv, est_hz,
        onset_tolerance=ONSET_TOLERANCE_S,
        pitch_tolerance=NOTE_PITCH_TOLERANCE_CENTS,
        offset_ratio=None,
    )

    return {'precision': float(p), 'recall': float(r), 'f1': float(f1),
            'n_ref': float(len(ref_notes)), 'n_est': float(len(est_notes))}


def score(ref: GroundTruth, est: Prediction,
          system: str = 'unknown') -> ScoreResult:
    """Score one prediction against one reference annotation."""
    return ScoreResult(
        track=ref.name,
        system=system,
        runtime_s=est.runtime_s,
        frame=_frame_scores(ref, est),
        note=_note_scores(ref, est),
        meta=dict(est.meta),
    )


def aggregate(results: List[ScoreResult]) -> Dict[str, float]:
    """Mean of every metric across tracks.

    An unweighted mean over tracks, not over frames: a long song should not
    outvote a short one when we are asking whether the system got better.
    """
    if not results:
        return {}

    out: Dict[str, float] = {}
    for key in results[0].frame:
        out[key] = float(np.mean([r.frame.get(key, 0.0) for r in results]))
    for key in results[0].note:
        out[f"note_{key}"] = float(np.mean([r.note.get(key, 0.0) for r in results]))
    out['runtime_s'] = float(np.sum([r.runtime_s for r in results]))
    out['n_tracks'] = float(len(results))
    return out


def format_table(results: List[ScoreResult],
                 columns: Optional[List[str]] = None) -> str:
    """Render per-track scores plus a MEAN row as a fixed-width table."""
    if not results:
        return "(no results)"

    columns = columns or ['Overall Accuracy', 'Raw Pitch Accuracy',
                          'Raw Chroma Accuracy', 'Voicing Recall',
                          'Voicing False Alarm', 'octave_error_rate']
    short = {'Overall Accuracy': 'OA', 'Raw Pitch Accuracy': 'RPA',
             'Raw Chroma Accuracy': 'RCA', 'Voicing Recall': 'VR',
             'Voicing False Alarm': 'VFA', 'octave_error_rate': 'OCT'}

    headers = ['track'] + [short.get(c, c) for c in columns] + ['noteF1', 'sec']
    rows: List[List[str]] = []
    for r in results:
        rows.append(
            [r.track[:28]]
            + [f"{r.frame.get(c, 0.0):.3f}" for c in columns]
            + [f"{r.note.get('f1', 0.0):.3f}", f"{r.runtime_s:.1f}"]
        )

    agg = aggregate(results)
    rows.append(
        ['MEAN']
        + [f"{agg.get(c, 0.0):.3f}" for c in columns]
        + [f"{agg.get('note_f1', 0.0):.3f}", f"{agg.get('runtime_s', 0.0):.1f}"]
    )

    widths = [max(len(headers[i]), max(len(row[i]) for row in rows))
              for i in range(len(headers))]
    line = '  '.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, '-' * len(line)]
    for row in rows[:-1]:
        out.append('  '.join(c.ljust(widths[i]) for i, c in enumerate(row)))
    out.append('-' * len(line))
    out.append('  '.join(c.ljust(widths[i]) for i, c in enumerate(rows[-1])))
    return '\n'.join(out)
