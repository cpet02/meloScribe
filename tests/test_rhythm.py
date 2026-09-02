"""Tests for rhythmic plausibility.

The load-bearing ones are the properties that were expensive to get right and
that a plausible-looking refactor would quietly break: the chance
normalisation, the refusal to quantise, and the grid's tolerance of triplets
and expressive timing.
"""

from __future__ import annotations

import numpy as np
import pytest

from meloscribe import rhythm
from meloscribe.eval.groundtruth import Note

BPM = 120.0
PERIOD = 60.0 / BPM


def grid(duration: float = 40.0) -> rhythm.BeatGrid:
    return rhythm.constant_tempo_grid(BPM, duration)


def metrical_notes(values, start_beat: float = 0.0, duty: float = 1.0):
    """Notes laid exactly on the grid, `values` being lengths in beats."""
    notes = []
    beat = start_beat
    for length in values:
        notes.append(Note(onset=beat * PERIOD,
                          offset=(beat + length * duty) * PERIOD, midi=60.0))
        beat += length
    return notes


EIGHTHS = [1.0, 0.5, 0.5, 1.0, 2.0, 0.5, 0.5, 1.0, 1.0, 2.0, 1.0, 0.5, 0.5]


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------

def test_deviation_is_zero_on_the_grid():
    report = rhythm.analyse(metrical_notes(EIGHTHS), grid())
    assert np.allclose([n.deviation_beats for n in report.notes], 0.0,
                       atol=1e-6)


def test_deviation_is_signed():
    """Early and late must be distinguishable, not folded to a distance."""
    early = [Note(onset=1.0 * PERIOD - 0.03, offset=2.0 * PERIOD, midi=60.0)]
    late = [Note(onset=1.0 * PERIOD + 0.03, offset=2.0 * PERIOD, midi=60.0)]
    assert grid().deviations([n.onset for n in early])[0] < 0
    assert grid().deviations([n.onset for n in late])[0] > 0


def test_triplets_are_on_the_grid():
    """A binary-only grid cannot express a triplet, and scored correct
    triplet material at 0.19 plausibility until the ternary positions were
    added. This is that regression."""
    report = rhythm.analyse(metrical_notes([1 / 3] * 9 + [1.0] * 6), grid())
    assert report.plausibility > 0.8
    assert report.on_grid_fraction == 1.0


def test_analysis_never_moves_a_note():
    """The grid observes; it must not quantise. Anything that starts snapping
    notes to the grid has to fail a test, not merely contradict a comment."""
    notes = metrical_notes(EIGHTHS)
    before = [(n.onset, n.offset) for n in notes]
    rhythm.analyse(notes, grid())
    assert [(n.onset, n.offset) for n in notes] == before


# --------------------------------------------------------------------------
# Chance normalisation
# --------------------------------------------------------------------------

def test_random_timing_scores_near_zero():
    """The whole metric rests on this: a chance-level input must score ~0.

    Averaged over draws, because at twenty-odd notes a single random draw has a
    standard deviation of about 0.1 - measured, not assumed.
    """
    rng = np.random.default_rng(7)
    scores = []
    for _ in range(12):
        onsets = np.sort(rng.uniform(1.0, 30.0, 24))
        lengths = np.exp(rng.uniform(np.log(0.1), np.log(2.0), 24)) * PERIOD
        notes = [Note(onset=float(o), offset=float(o + d), midi=60.0)
                 for o, d in zip(onsets, lengths)]
        scores.append(rhythm.analyse(notes, grid()).plausibility)
    assert np.mean(scores) < 0.2


def test_metrical_beats_chance():
    """Chance is high enough that ignoring it would be fatal to the metric."""
    assert rhythm._onset_chance(20, rhythm.ONSET_SIGMA_BEATS) > 0.5


def test_clean_beats_corrupted():
    clean = metrical_notes(EIGHTHS)
    rng = np.random.default_rng(3)
    shifted = [Note(onset=n.onset + float(rng.uniform(-0.25, 0.25)) * PERIOD,
                    offset=n.offset, midi=n.midi) for n in clean]
    assert (rhythm.analyse(clean, grid()).plausibility
            > rhythm.analyse(shifted, grid()).plausibility + 0.3)


# --------------------------------------------------------------------------
# Expressive timing must survive
# --------------------------------------------------------------------------

def test_systematic_lag_is_forgiven():
    """A singer consistently behind the band is not a transcription error."""
    lag = 0.4 * rhythm.ONSET_SIGMA_BEATS * PERIOD
    notes = [Note(onset=n.onset + lag, offset=n.offset + lag, midi=n.midi)
             for n in metrical_notes(EIGHTHS)]
    assert rhythm.analyse(notes, grid()).plausibility > 0.8


def test_offset_cannot_rescue_scattered_onsets():
    """The global offset is one scalar; it must not be able to fit noise."""
    rng = np.random.default_rng(11)
    onsets = np.sort(rng.uniform(1.0, 30.0, 24))
    notes = [Note(onset=float(o), offset=float(o) + PERIOD, midi=60.0)
             for o in onsets]
    report = rhythm.analyse(notes, grid())
    assert report.onset_score < 0.35


def test_short_releases_are_forgiven():
    """Notes released early are normal singing, not implausible rhythm.

    0.88 rather than 0.75 deliberately: three quarters of a beat *is* a dotted
    eighth, so a 25% shortening is genuinely ambiguous with a real note value
    and the measure cannot be asked to see through it.
    """
    notes = metrical_notes(EIGHTHS, duty=0.88)
    assert rhythm.analyse(notes, grid()).duration_score > 0.8


# --------------------------------------------------------------------------
# Trusting the tempo
# --------------------------------------------------------------------------

def test_unreliable_grid_is_never_flagged():
    """A bad tempo must not masquerade as a bad transcription."""
    bad = rhythm.constant_tempo_grid(BPM, 40.0, confidence=0.0)
    report = rhythm.analyse(metrical_notes(EIGHTHS), bad)
    assert not report.reliable
    assert not report.flagged


def test_too_few_notes_is_not_assessed():
    report = rhythm.analyse(metrical_notes([1.0, 1.0, 0.5]), grid())
    assert not report.reliable
    assert not report.flagged


def test_scattered_transcription_is_flagged():
    rng = np.random.default_rng(5)
    onsets = np.sort(rng.uniform(1.0, 30.0, 30))
    notes = [Note(onset=float(o),
                  offset=float(o + rng.uniform(0.1, 0.9)), midi=60.0)
             for o in onsets]
    report = rhythm.analyse(notes, grid())
    assert report.reliable and report.flagged


def test_implausible_bpm_lowers_confidence():
    envelope = np.ones(2000)
    frames = np.arange(0, 2000, 5.0)   # ~40 BPM at the default hop
    confidence, warnings = rhythm._tempo_diagnostics(envelope, frames, 40.0)
    assert confidence < rhythm.MIN_TEMPO_CONFIDENCE
    assert any('range' in w for w in warnings)


def test_uneven_beats_lower_confidence():
    rng = np.random.default_rng(2)
    frames = np.cumsum(rng.uniform(10.0, 40.0, 40))
    confidence, warnings = rhythm._tempo_diagnostics(np.ones(2000), frames,
                                                     120.0)
    assert confidence < rhythm.MIN_TEMPO_CONFIDENCE
    assert any('uneven' in w for w in warnings)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def test_grid_round_trips():
    original = grid(20.0)
    restored = rhythm.BeatGrid.from_dict(original.to_dict())
    assert restored.bpm == pytest.approx(original.bpm)
    assert np.allclose(restored.beat_times, original.beat_times, atol=1e-3)
    assert restored.reliable == original.reliable


def test_report_dict_is_json_safe():
    import json
    report = rhythm.analyse(metrical_notes(EIGHTHS), grid())
    assert json.loads(json.dumps(report.to_dict()))['reliable'] is True


def test_accepts_engine_notes():
    """`analyse` must take a TranscribedNote as well as an eval Note."""
    from meloscribe.pitch.engine import TranscribedNote
    notes = [TranscribedNote(midi=60, start=n.onset, end=n.offset,
                             confidence=0.9) for n in metrical_notes(EIGHTHS)]
    assert rhythm.analyse(notes, grid()).plausibility > 0.8
