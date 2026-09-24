"""Tests for the real-corpus layout and converters.

Each converter encodes a corpus convention (column order, frame timing,
tuning) that a silent mistake would turn into a plausible-looking but wrong
score, so each is pinned against a tiny fabricated copy of that corpus.
"""

import json

import numpy as np
import pytest
import soundfile as sf

from meloscribe.eval.datasets import (convert_ikala, convert_medleydb_melody,
                                      convert_tonas, convert_vocadito,
                                      is_auxiliary, read_notes_csv,
                                      write_notes_csv)
from meloscribe.eval.groundtruth import Note, load_csv_f0
from meloscribe.eval.runner import load_dataset

SR = 8000


def _wav(path, channels=1, seconds=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((int(SR * seconds), channels))
    if channels == 2:
        data[:, 0], data[:, 1] = 0.25, 0.5    # left = music, right = voice
    sf.write(path, data, SR)


def test_notes_csv_roundtrip(tmp_path):
    notes = [Note(0.5, 0.9, 64.0), Note(0.1, 0.4, 60.5)]
    write_notes_csv(tmp_path / 'x.notes.csv', notes)
    back = read_notes_csv(tmp_path / 'x.notes.csv')
    assert [(n.onset, n.offset, n.midi) for n in back] == [(0.1, 0.4, 60.5),
                                                           (0.5, 0.9, 64.0)]


def test_auxiliary_csvs_are_not_tracks(tmp_path):
    assert is_auxiliary(tmp_path / 'song.notes.csv')
    assert is_auxiliary(tmp_path / 'song.notes.A2.csv')
    assert not is_auxiliary(tmp_path / 'song.csv')


def test_load_dataset_attaches_reference_notes(tmp_path):
    _wav(tmp_path / 'song.wav')
    (tmp_path / 'song.csv').write_text('0.00,0\n0.01,440\n')
    write_notes_csv(tmp_path / 'song.notes.csv', [Note(0.0, 0.3, 69.0)])
    write_notes_csv(tmp_path / 'song.notes.A2.csv', [Note(0.0, 0.2, 69.0)])
    (tmp_path / 'song.alto.csv').write_text('0.00,220\n')

    truths = load_dataset(tmp_path)

    assert [t.name for t in truths] == ['song']
    assert [(n.onset, n.offset, n.midi) for n in truths[0].notes] == [(0.0, 0.3, 69.0)]


def test_vocadito_notes_are_onset_hz_duration(tmp_path):
    src = tmp_path / 'vocadito'
    _wav(src / 'Audio' / 'vocadito_3.wav')
    (src / 'Annotations' / 'F0').mkdir(parents=True)
    (src / 'Annotations' / 'Notes').mkdir(parents=True)
    (src / 'Annotations' / 'F0' / 'vocadito_3_f0.csv').write_text('0.0,0.0\n0.0058,220.0\n')
    (src / 'Annotations' / 'Notes' / 'vocadito_3_notesA1.csv').write_text('0.5,440.0,0.25\n')
    (src / 'Annotations' / 'Notes' / 'vocadito_3_notesA2.csv').write_text('0.5,880.0,0.25\n')

    assert convert_vocadito(src, tmp_path / 'out') == ['vocadito_3']

    notes = read_notes_csv(tmp_path / 'out' / 'vocadito_3.notes.csv')
    assert (notes[0].onset, notes[0].offset) == (0.5, 0.75)
    assert notes[0].midi == pytest.approx(69.0)
    assert read_notes_csv(tmp_path / 'out' / 'vocadito_3.notes.A2.csv')[0].midi \
        == pytest.approx(81.0)
    truth = load_csv_f0(tmp_path / 'out' / 'vocadito_3.csv')
    assert truth.freqs.tolist() == [0.0, 220.0]


def test_ikala_splits_channels_and_centres_32ms_frames(tmp_path):
    src = tmp_path / 'iKala'
    _wav(src / 'Wavfile' / '10161_chorus.wav', channels=2)
    (src / 'PitchLabel').mkdir(parents=True)
    (src / 'PitchLabel' / '10161_chorus.pv').write_text('0 \n69 \n')

    names = convert_ikala(src, tmp_path / 'out')

    assert sorted(names) == ['10161_chorus_mix', '10161_chorus_voice']
    voice, _ = sf.read(tmp_path / 'out' / '10161_chorus_voice.wav')
    mix, _ = sf.read(tmp_path / 'out' / '10161_chorus_mix.wav')
    assert voice[0] == pytest.approx(0.5)          # right channel
    assert mix[0] == pytest.approx(0.75)           # left + right
    truth = load_csv_f0(tmp_path / 'out' / '10161_chorus_mix.csv')
    assert truth.times.tolist() == pytest.approx([0.016, 0.048])
    assert truth.freqs.tolist() == pytest.approx([0.0, 440.0])


def test_tonas_applies_tuning_offset(tmp_path):
    src = tmp_path / 'TONAS'
    _wav(src / 'Deblas' / '01-D_X.wav')
    (src / 'Deblas' / '01-D_X.f0.Corrected').write_text(
        '0.197 0.1 0.000 0.000\n0.209 0.1 143.9 379.3\n')
    (src / 'Deblas' / '01-D_X.notes.Corrected').write_text(
        '50.000000\n0.2, 0.4, 66.00, 0.01\n')

    assert convert_tonas(src, tmp_path / 'out') == ['01-D_X']

    note = read_notes_csv(tmp_path / 'out' / '01-D_X.notes.csv')[0]
    assert (note.onset, note.offset) == pytest.approx((0.2, 0.6))
    assert note.midi == pytest.approx(66.5)         # +50 cents of tuning
    truth = load_csv_f0(tmp_path / 'out' / '01-D_X.csv')
    assert truth.freqs.tolist() == pytest.approx([0.0, 379.3])  # corrected column


def test_medleydb_skips_instrumentals(tmp_path):
    src = tmp_path / 'mdb'
    for track in ('Sung_Song', 'Only_Strings'):
        _wav(src / 'audio' / f'{track}_MIX.wav')
        (src / 'melody2').mkdir(parents=True, exist_ok=True)
        (src / 'melody2' / f'{track}_MELODY2.csv').write_text('0.0,0.0\n0.0058,220.0\n')
    (src / 'medleydb_melody_metadata.json').write_text(json.dumps({
        'Sung_Song': {'is_instrumental': False},
        'Only_Strings': {'is_instrumental': True}}))

    assert convert_medleydb_melody(src, tmp_path / 'out') == ['Sung_Song']
