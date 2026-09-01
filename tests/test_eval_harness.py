"""Tests for the scoring harness itself.

The harness is the instrument every later accuracy claim rests on, so its own
correctness cannot be taken on faith.
"""

import numpy as np
import pytest

from meloscribe.eval.groundtruth import (GroundTruth, Note, Prediction,
                                         f0_to_notes, hz_to_midi, midi_to_hz,
                                         notes_to_f0, resample_f0)
from meloscribe.eval.metrics import aggregate, score


def test_midi_hz_roundtrip():
    for midi in (36.0, 60.0, 69.0, 84.5):
        assert hz_to_midi(np.array([midi_to_hz(midi)]))[0] == pytest.approx(midi)


def test_hz_to_midi_keeps_unvoiced_at_zero():
    assert hz_to_midi(np.array([0.0, 440.0])).tolist() == [0.0, 69.0]


def test_notes_to_f0_rests_are_unvoiced():
    times, freqs = notes_to_f0([Note(0.0, 0.5, 60), Note(1.0, 1.5, 62)])
    assert freqs[10] > 0          # inside the first note
    assert freqs[75] == 0.0       # inside the rest
    assert freqs[110] > 0         # inside the second note


def test_f0_to_notes_recovers_original_notes():
    original = [Note(0.2, 0.7, 60), Note(0.8, 1.3, 64), Note(1.5, 2.0, 67)]
    notes = f0_to_notes(*notes_to_f0(original))

    assert len(notes) == len(original)
    for got, want in zip(notes, original):
        assert got.midi == pytest.approx(want.midi, abs=0.01)
        assert got.onset == pytest.approx(want.onset, abs=0.02)


def test_f0_to_notes_drops_sub_threshold_blips():
    times = np.arange(100) * 0.01
    freqs = np.zeros(100)
    freqs[50:52] = 440.0  # a 20ms blip, below the 50ms floor
    assert f0_to_notes(times, freqs, min_duration=0.05) == []


def test_resample_past_end_is_unvoiced():
    times, freqs = notes_to_f0([Note(0.0, 1.0, 60)])
    out = resample_f0(times, freqs, np.array([0.5, 5.0]))
    assert out[0] > 0
    assert out[1] == 0.0


def _truth():
    notes = [Note(0.2, 0.7, 60), Note(0.8, 1.3, 64)]
    times, freqs = notes_to_f0(notes, duration=2.0)
    return GroundTruth(name='t', times=times, freqs=freqs, notes=notes)


def test_perfect_prediction_scores_one():
    truth = _truth()
    result = score(truth, Prediction(name='t', times=truth.times,
                                     freqs=truth.freqs, notes=truth.notes))
    assert result.frame['Overall Accuracy'] == pytest.approx(1.0)
    assert result.note['f1'] == pytest.approx(1.0)
    assert result.frame['octave_error_rate'] == 0.0


def test_octave_error_is_detected_and_separated_from_chroma():
    truth = _truth()
    result = score(truth, Prediction(name='t', times=truth.times,
                                     freqs=truth.freqs * 2.0))

    # An octave up: right pitch class, wrong pitch. The diagnostic must
    # separate these, since one is far more fixable than the other.
    assert result.frame['octave_error_rate'] == pytest.approx(1.0)
    assert result.frame['Raw Chroma Accuracy'] == pytest.approx(1.0)
    assert result.frame['Raw Pitch Accuracy'] == pytest.approx(0.0)


def test_silent_prediction_scores_zero_not_crash():
    truth = _truth()
    result = score(truth, Prediction(name='t', times=truth.times,
                                     freqs=np.zeros_like(truth.freqs)))

    # Predicting silence everywhere still earns credit for the frames that
    # really are unvoiced, so OA lands at exactly the rest fraction rather
    # than at zero. Worth pinning: it is why OA alone can flatter a system
    # that simply refuses to commit.
    assert result.frame['Raw Pitch Accuracy'] == 0.0
    assert result.frame['Voicing Recall'] == 0.0
    assert result.frame['Overall Accuracy'] == pytest.approx(
        1.0 - truth.voiced_fraction, abs=0.01)
    assert result.note['f1'] == 0.0


def test_mismatched_times_and_freqs_rejected():
    with pytest.raises(ValueError, match='length mismatch'):
        GroundTruth(name='bad', times=np.zeros(5), freqs=np.zeros(3))


def test_aggregate_averages_over_tracks():
    truth = _truth()
    good = score(truth, Prediction(name='t', times=truth.times,
                                   freqs=truth.freqs), system='s')
    bad = score(truth, Prediction(name='t', times=truth.times,
                                  freqs=np.zeros_like(truth.freqs)), system='s')
    agg = aggregate([good, bad])

    assert agg['n_tracks'] == 2
    assert 0.0 < agg['Overall Accuracy'] < 1.0
