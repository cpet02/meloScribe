"""Smoke tests for the stress harness (`meloscribe.eval.stress`), plus
regression tests for the input-robustness bugs it found.

Seconds and no network: the harness tests run the engine on basic-pitch alone
over a two-second phrase, and the regression tests exercise loading and the
CLI with at most that one small model.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import soundfile as sf

from meloscribe import audio as audio_mod
from meloscribe.eval import stress
from meloscribe.eval.groundtruth import Note

FAST = ('basic_pitch',)


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


# --------------------------------------------------------------------------
# The harness itself
# --------------------------------------------------------------------------

def test_easy_sweep_point_scores_perfectly(tmp_path):
    rows = stress.run_point('vibrato_depth', 0, tmp_path, systems=('ensemble',),
                            quick=True, voters=FAST)
    ens = rows[0]
    assert (ens['system'], ens['axis'], ens['n_ref']) == ('ensemble', 'vibrato_depth', 4)
    # The per-voter numbers come from re-decoding captured voter outputs;
    # that is only meaningful if re-decoding reproduces the engine exactly.
    assert ens['redecode_consistent'] is True
    assert ens['f1'] >= 0.99, ens


def test_scoring_separates_octave_errors_from_fragmentation():
    ref = [Note(0.0, 1.0, 60.0)]
    est = [Note(0.0, 0.5, 72.0), Note(0.5, 1.0, 72.0)]
    scores = stress.note_scores(ref, est)
    assert scores['f1'] == 0.0
    assert scores['oct_note'] == 1.0
    assert scores['frag'] == 2.0


def test_silence_fuzz_case_yields_no_notes(tmp_path):
    rows = stress.run_fuzz_case('silence', tmp_path, entries=('engine',),
                                voters=FAST)
    assert rows[0]['verdict'] == 'ok', rows[0]


# --------------------------------------------------------------------------
# Regression tests for bugs the stress run found
# --------------------------------------------------------------------------

def test_non_finite_samples_are_repaired_not_fatal(tmp_path):
    """One NaN sample used to kill the run inside librosa.resample with
    'Audio buffer is not finite everywhere'."""
    x, _ = stress._fuzz_voice()
    x[1000], x[2000] = np.nan, np.inf
    path = tmp_path / 'nan.wav'
    sf.write(str(path), x, stress.SR, subtype='FLOAT')
    with pytest.warns(UserWarning, match='non-finite'):
        audio = audio_mod.load(path, sr=audio_mod.ANALYSIS_SR)
    assert np.isfinite(audio.samples).all()
    assert audio.repairs


def test_engine_survives_non_finite_samples(tmp_path):
    """basic-pitch reads the file itself, so repairing the array alone is not
    enough: it must be handed the repaired audio too."""
    from meloscribe.pitch.engine import EngineSettings, PitchEngine
    x, _ = stress._fuzz_voice()
    x[len(x) // 2] = np.nan
    path = tmp_path / 'nan.wav'
    sf.write(str(path), x, stress.SR, subtype='FLOAT')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        notes = PitchEngine(EngineSettings(voters=FAST)).transcribe(path).notes
    assert {57, 60, 64, 69} <= {n.midi for n in notes}


def test_engine_survives_a_file_name_stdout_cannot_encode(tmp_path, monkeypatch):
    """basic-pitch prints the path it reads to stdout. Redirected on Windows,
    stdout is strict cp1252, so a song named in Chinese crashed the engine -
    under a web server logging to a file, say, rather than the CLI, which
    moves library output to stderr itself."""
    import io
    import sys
    from meloscribe.pitch.engine import EngineSettings, PitchEngine
    x, _ = stress._fuzz_voice()
    path = tmp_path / 'Café del Mar – 你好.wav'
    sf.write(str(path), x, stress.SR)
    monkeypatch.setattr(sys, 'stdout', io.TextIOWrapper(io.BytesIO(),
                                                        encoding='cp1252'))
    notes = PitchEngine(EngineSettings(voters=FAST)).transcribe(path).notes
    assert {57, 60, 64, 69} <= {n.midi for n in notes}


def test_phase_inverted_stereo_does_not_cancel(tmp_path):
    """L = v, R = -v sums to digital silence in a plain downmix."""
    x, _ = stress._fuzz_voice()
    path = tmp_path / 'inverted.wav'
    sf.write(str(path), np.stack([x, -x], axis=1), stress.SR, subtype='FLOAT')
    with pytest.warns(UserWarning, match='cancel'):
        audio = audio_mod.load(path)
    assert _rms(audio.samples) > 0.5 * _rms(x)


def test_ordinary_stereo_is_still_averaged(tmp_path):
    x, _ = stress._fuzz_voice()
    path = tmp_path / 'stereo.wav'
    sf.write(str(path), np.stack([x, 0.5 * x], axis=1), stress.SR, subtype='FLOAT')
    with warnings.catch_warnings():
        warnings.simplefilter('error')      # no repair warning on normal input
        audio = audio_mod.load(path)
    assert np.allclose(audio.samples, 0.75 * x, atol=1e-6)
    assert not audio.repairs


@pytest.mark.parametrize('content', [b'', b'garbage'])
def test_undecodable_or_empty_input_raises_a_clear_error(tmp_path, content):
    path = tmp_path / 'broken.mp3'
    if content:
        path.write_bytes(np.random.default_rng(0).bytes(20_000))
    else:
        sf.write(str(path), np.zeros(0), 16000, format='WAV')
    with pytest.raises(audio_mod.AudioLoadError, match='broken.mp3'):
        audio_mod.load(path)


def test_cli_reports_undecodable_input_without_a_traceback(tmp_path, capsys):
    from meloscribe import cli
    path = tmp_path / 'garbage.mp3'
    path.write_bytes(np.random.default_rng(1).bytes(20_000))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rc = cli.main([str(path), '--vocals-only', '--no-lyrics', '--quiet'])
    assert rc == 1
    assert 'garbage.mp3' in capsys.readouterr().err


def test_cli_stdout_stays_pure_json_when_a_library_prints(tmp_path, monkeypatch,
                                                           capsys):
    """basic-pitch prints 'Predicting MIDI for ...' to stdout, which made
    `meloscribe x.wav --format json > notes.json` write invalid JSON."""
    from meloscribe import cli
    from meloscribe.pipeline import TranscriptionOutput

    class ChattyPipeline:
        def run(self, request, progress=None):
            print(f"Predicting MIDI for {request.input_path}...")
            return TranscriptionOutput(notes=[])

    monkeypatch.setattr(cli, 'Pipeline', ChattyPipeline)
    path = tmp_path / 'a.wav'
    sf.write(str(path), np.zeros(1600), 16000)
    assert cli.main([str(path), '--vocals-only', '--no-lyrics', '--quiet',
                     '--format', 'json']) == 0
    assert json.loads(capsys.readouterr().out)['notes'] == []


@pytest.mark.parametrize('seconds', [0.002, 0.07])
def test_engine_survives_clips_shorter_than_a_note(tmp_path, seconds):
    """A 2 ms file crashed inside basic-pitch's note decoding with
    'zero-size array to reduction operation maximum'."""
    from meloscribe.pitch.engine import EngineSettings, PitchEngine
    x, _ = stress._fuzz_voice()
    path = tmp_path / 'tiny.wav'
    start = int(0.5 * stress.SR)
    sf.write(str(path), x[start:start + int(seconds * stress.SR)], stress.SR)
    result = PitchEngine(EngineSettings(voters=FAST)).transcribe(path)
    assert len(result.notes) <= 1
