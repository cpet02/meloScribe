"""Smoke test for the realistic sung-vocal benchmark.

Generates one 3-second case with the analytic formant voice (so no recording
and no network are needed) and checks the two properties everything else rests
on: the harness scores its own truth as perfect, and the truth describes the
audio that was actually rendered.
"""

import numpy as np
import pytest

# The generator needs `pip install pyworld`; scoring needs mir_eval. pysptk is
# NOT needed here: this test uses the analytic formant voice, not the CMU
# ARCTIC recording, so it runs with no network and no pysptk.
pytest.importorskip('pyworld')
pytest.importorskip('mir_eval')
pytest.importorskip('librosa')
_sf = pytest.importorskip('soundfile')
if 'MP3' not in _sf.available_formats():
    pytest.skip('libsndfile without MP3 support (soundfile>=0.12 wheels bundle it)',
                allow_module_level=True)

from meloscribe.eval import realistic as R  # noqa: E402


@pytest.fixture(scope='module')
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp('realistic')
    return R.generate_case(R.SMOKE_CASE, out, voice='formant')


def test_case_is_written_and_reloads(smoke):
    for name in ('mix.mp3', 'vocals.wav', 'proxy.wav', 'truth.json'):
        assert (smoke.case_dir / name).exists(), name
    assert 2.5 <= smoke.duration <= 4.0
    assert len(smoke.notes) >= 2
    again = R.load_truths(smoke.case_dir.parent, [smoke.name])[0]
    assert [n.midi for n in again.notes] == [n.midi for n in smoke.notes]
    assert np.allclose(again.freqs, smoke.freqs, atol=0.01)


def test_oracle_scores_one(smoke):
    row = R.score_case(smoke, R.oracle_prediction(smoke))
    for key in ('note_f1', 'note_f1_off', 'OA', 'RPA'):
        assert row[key] == pytest.approx(1.0), key
    # Not 1.0 by design: the sung contour itself leaves the intended note's
    # 50-cent band during scoops, glides and falls.
    assert row['RPA_note'] > 0.9
    assert row['spurious_rate'] == 0.0
    assert row['fragmentation'] == 1.0
    assert row['merge_rate'] == 0.0
    assert row['phantom_rate'] == 0.0
    assert row['errors'] == {}


def test_truth_f0_is_what_was_sung(smoke):
    """WORLD must reproduce the imposed f0, or the truth is fiction."""
    import librosa
    y, sr = librosa.load(str(smoke.audio('clean')), sr=16000)
    f0 = librosa.yin(y, fmin=100, fmax=1000, sr=sr, frame_length=1024, hop_length=160)
    n = min(len(f0), len(smoke.freqs))
    held = smoke.freqs[:n] > 0
    # Frames well inside notes only: onsets, consonants and glides are where
    # any tracker (yin included) is legitimately unsure.
    inner = held & np.roll(held, 5) & np.roll(held, -5)
    cents = 1200 * np.abs(np.log2(f0[:n][inner] / smoke.freqs[:n][inner]))
    assert np.median(cents) < 20
