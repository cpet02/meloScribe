"""Tests for the real-corpus layout and converters.

Each converter encodes a corpus convention (column order, frame timing,
tuning) that a silent mistake would turn into a plausible-looking but wrong
score, so each is pinned against a tiny fabricated copy of that corpus.
"""

import json
import warnings

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


def test_load_dataset_keeps_tracks_with_dots_in_their_names(tmp_path):
    # A hand-made folder: the dot is part of the track's name, not a sign
    # that the CSV annotates some other track.
    for name in ('01. Intro', 'take 1.5'):
        _wav(tmp_path / f'{name}.wav')
        (tmp_path / f'{name}.csv').write_text('0.00,0\n0.01,440\n')
    write_notes_csv(tmp_path / '01. Intro.notes.csv', [Note(0.0, 0.3, 69.0)])

    truths = load_dataset(tmp_path)

    assert [t.name for t in truths] == ['01. Intro', 'take 1.5']
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


def _tonas(tmp_path, stored_midi):
    """A TONAS track sung 43 cents sharp of A440: f0 379.3 Hz (MIDI 66.43)
    over its one note, as in mirdata's own sample file."""
    src = tmp_path / 'TONAS'
    _wav(src / 'Deblas' / '01-D_X.wav')
    (src / 'Deblas' / '01-D_X.f0.Corrected').write_text(
        '0.197 0.1 0.000 0.000\n0.209 0.1 143.9 379.3\n'
        '0.300 0.1 143.9 379.3\n0.500 0.1 143.9 379.3\n')
    (src / 'Deblas' / '01-D_X.notes.Corrected').write_text(
        f'43.000000\n0.216667, 0.433333, {stored_midi}, 0.018007\n')
    return src


def test_tonas_notes_already_include_the_tuning(tmp_path):
    # The MIDI column carries the 43 cents itself (66.43, and the f0 agrees);
    # adding the first line's tuning again, as mirdata does, gave 66.86.
    src = _tonas(tmp_path, '66.430000')
    with warnings.catch_warnings():
        warnings.simplefilter('error')      # the f0 check must stay quiet
        assert convert_tonas(src, tmp_path / 'out') == ['01-D_X']

    note = read_notes_csv(tmp_path / 'out' / '01-D_X.notes.csv')[0]
    assert (note.onset, note.offset) == pytest.approx((0.216667, 0.65))
    assert note.midi == pytest.approx(66.43)
    truth = load_csv_f0(tmp_path / 'out' / '01-D_X.csv')
    assert truth.freqs.tolist() == pytest.approx([0.0, 379.3, 379.3, 379.3])


def test_tonas_notes_off_their_own_f0_are_flagged(tmp_path):
    # A file in a different convention (notes in whole semitones of the
    # singer's tuning) would score against pitches 43 cents off: say so.
    with pytest.warns(UserWarning, match=r'\+43 cents from the f0'):
        convert_tonas(_tonas(tmp_path, '66.000000'), tmp_path / 'out')


def test_medleydb_refuses_the_multi_line_melody_definition(tmp_path):
    # MELODY3 has a column per melodic line; only the first was kept.
    with pytest.raises(ValueError, match='not a single line'):
        convert_medleydb_melody(tmp_path / 'mdb', tmp_path / 'out', definition=3)


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
