"""Note spelling: each note named as the key it is written in calls for.

A player reading in Eb major expects Bb, Eb and Ab, not A#, D# and G#; one
reading in D minor expects Bb and C#, not A# or Db. After transposition the
key they read in is not the key that was sung. Spelling is presentation only:
it must never move a note's MIDI number.

Like the unit suite, nothing here needs models, network or audio. The pipeline
tests supply the key estimate and the sung notes themselves.
"""

import csv
import io
import json
import sys
import types
from dataclasses import replace

import pytest

from meloscribe import cli
from meloscribe.key import SPELLINGS, KeyEstimate, note_names, note_spelling
from meloscribe.lyrics.service import LyricsMode
from meloscribe.output import (format_csv, format_json, format_leadsheet,
                               format_lrc, format_table, write_midi)
from meloscribe.pipeline import Pipeline, TranscriptionRequest
from meloscribe.pitch.engine import (PitchEngine, TranscribedNote,
                                     TranscriptionResult, midi_to_name)

ALTO_SAX = 9     # Eb instrument: written a major sixth above concert pitch
TENOR_SAX = 2    # Bb instrument: a major second (plus an octave) above

SHARPS = SPELLINGS['sharp']
D_MINOR = KeyEstimate(tonic=2, is_major=False, confidence=0.9)


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------

@pytest.mark.parametrize('tonic,is_major,name,spelling', [
    (0, True, 'C major', 'sharp'),      # no accidentals: sharps, as before
    (1, True, 'Db major', 'flat'),      # five flats beat C# major's seven sharps
    (2, True, 'D major', 'sharp'),
    (3, True, 'Eb major', 'flat'),
    (4, True, 'E major', 'sharp'),
    (5, True, 'F major', 'flat'),
    (6, True, 'Gb major', 'flat'),      # the six-six tie, settled as flats
    (7, True, 'G major', 'sharp'),
    (8, True, 'Ab major', 'flat'),
    (9, True, 'A major', 'sharp'),
    (10, True, 'Bb major', 'flat'),
    (11, True, 'B major', 'sharp'),     # five sharps beat Cb major's seven flats
    (0, False, 'C minor', 'flat'),
    (1, False, 'C# minor', 'sharp'),
    (2, False, 'D minor', 'flat'),
    (3, False, 'Eb minor', 'flat'),     # the tie again, via its relative Gb
    (4, False, 'E minor', 'sharp'),
    (5, False, 'F minor', 'flat'),
    (6, False, 'F# minor', 'sharp'),
    (7, False, 'G minor', 'flat'),
    (8, False, 'G# minor', 'sharp'),
    (9, False, 'A minor', 'sharp'),     # no accidentals: sharps, as before
    (10, False, 'Bb minor', 'flat'),
    (11, False, 'B minor', 'sharp'),
])
def test_every_key_is_named_and_spelled_by_its_signature(tonic, is_major,
                                                         name, spelling):
    key = KeyEstimate(tonic=tonic, is_major=is_major, confidence=0.9)
    assert key.name == name
    assert key.spelling == spelling
    assert note_spelling(key) == spelling


def _signature_table(key):
    """The flat or sharp table, with Cb for the B that six flats flatten."""
    names = list(SPELLINGS[key.spelling])
    if key.fifths == -6:
        names[11] = 'Cb'
    return names


def test_major_keys_name_every_pitch_by_their_signature():
    for tonic in range(12):
        key = KeyEstimate(tonic=tonic, is_major=True, confidence=0.9)
        assert list(key.pitch_names) == _signature_table(key), key.name


@pytest.mark.parametrize('is_major', [True, False])
@pytest.mark.parametrize('tonic', range(12))
def test_every_key_spells_its_own_scale_with_seven_letters(tonic, is_major):
    """A signature sharpens or flattens whole letters, so a key's own scale
    uses each letter once. Six flats (Gb major, Eb minor) called their Cb
    'B' - B natural, under a signature that flattens every B."""
    key = KeyEstimate(tonic=tonic, is_major=is_major, confidence=0.9)
    steps = (0, 2, 4, 5, 7, 9, 11) if is_major else (0, 2, 3, 5, 7, 8, 10)
    letters = sorted(key.pitch_names[(tonic + s) % 12][0] for s in steps)
    assert letters == list('ABCDEFG'), key.name


def test_six_flats_write_cb_an_octave_up_from_its_b():
    for key in (KeyEstimate(tonic=6, is_major=True, confidence=0.9),
                KeyEstimate(tonic=3, is_major=False, confidence=0.9)):
        assert midi_to_name(71, key.pitch_names) == 'Cb5', key.name
        assert midi_to_name(59, key.pitch_names) == 'Cb4', key.name


@pytest.mark.parametrize('tonic,leading_tone', [
    (0, 'B'), (1, 'B#'), (2, 'C#'), (3, 'D'), (4, 'D#'), (5, 'E'),
    (6, 'E#'), (7, 'F#'), (8, 'F##'), (9, 'G#'), (10, 'A'), (11, 'A#'),
])
def test_a_minor_key_writes_its_leading_tone_as_the_raised_seventh(
        tonic, leading_tone):
    """Whatever the signature alone would call it - Db in D minor, F in F#
    minor, both of which read as wrong notes. Nothing else changes."""
    key = KeyEstimate(tonic=tonic, is_major=False, confidence=0.9)
    leading = (tonic - 1) % 12
    assert key.pitch_names[leading] == leading_tone
    assert [n for pc, n in enumerate(key.pitch_names) if pc != leading] == \
        [n for pc, n in enumerate(_signature_table(key)) if pc != leading]


def test_d_and_g_minor_keep_their_flats_beside_the_raised_seventh():
    d_minor = D_MINOR.pitch_names
    assert (d_minor[1], d_minor[10]) == ('C#', 'Bb')
    g_minor = KeyEstimate(tonic=7, is_major=False, confidence=0.9).pitch_names
    assert (g_minor[6], g_minor[10], g_minor[3]) == ('F#', 'Bb', 'Eb')


@pytest.mark.parametrize('concert,transpose,written,spelling', [
    # The case that motivated this: concert Gb major on alto sax.
    ((6, True), ALTO_SAX, 'Eb major', 'flat'),
    # The written key decides, and transposing can flip it either way.
    ((10, True), ALTO_SAX, 'G major', 'sharp'),     # concert Bb: flats -> sharps
    ((4, True), ALTO_SAX, 'Db major', 'flat'),      # concert E: sharps -> flats
    ((3, True), TENOR_SAX, 'F major', 'flat'),
    ((10, True), TENOR_SAX, 'C major', 'sharp'),
    ((0, False), ALTO_SAX, 'A minor', 'sharp'),     # concert C minor
    ((7, True), -ALTO_SAX, 'Bb major', 'flat'),     # an alto part in G, to concert
])
def test_the_written_key_decides_the_spelling(concert, transpose, written,
                                              spelling):
    key = KeyEstimate(tonic=concert[0], is_major=concert[1], confidence=0.9)
    assert key.transposed(transpose).name == written
    assert note_spelling(key, transpose) == spelling
    assert note_names(key, transpose) == key.transposed(transpose).pitch_names


def test_the_leading_tone_follows_the_written_key():
    """Concert F minor's leading tone is a plain E; on alto sax it is written
    in D minor, where the leading tone needs its sharp."""
    f_minor = KeyEstimate(tonic=5, is_major=False, confidence=0.9)
    assert note_names(f_minor, ALTO_SAX) == D_MINOR.pitch_names
    assert note_names(f_minor, ALTO_SAX)[1] == 'C#'


def test_an_uncertain_key_keeps_sharps():
    """A key too weak to bias the decoder must not choose between Bb and A#
    either. The gate is the prior's own, so the two cannot disagree about
    whether the key is usable - and the flag and the table share it."""
    assert note_spelling(None, ALTO_SAX) == 'sharp'
    assert note_names(None, ALTO_SAX) == SHARPS

    outcomes = set()
    for confidence in (0.0, 0.2, 0.5, 0.54, 0.55, 0.7, 1.0):
        d_minor = KeyEstimate(tonic=2, is_major=False, confidence=confidence)
        trusted = d_minor.as_prior() is not None
        assert note_spelling(d_minor) == ('flat' if trusted else 'sharp')
        assert note_names(d_minor) == (d_minor.pitch_names if trusted
                                       else SHARPS)
        outcomes.add(trusted)
    assert outcomes == {True, False}, 'both sides of the gate must be exercised'


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------

def test_names_change_never_the_pitch():
    assert midi_to_name(70) == 'A#4'
    assert midi_to_name(70, D_MINOR.pitch_names) == 'Bb4'
    assert midi_to_name(61, D_MINOR.pitch_names) == 'C#4'
    assert midi_to_name(60, D_MINOR.pitch_names) == midi_to_name(60) == 'C4'

    note = TranscribedNote(70, 0.0, 0.5, 0.9, pitch_names=D_MINOR.pitch_names)
    assert (note.name, note.midi) == ('Bb4', 70)
    assert (note.to_dict()['note'], note.to_dict()['midi']) == ('Bb4', 70)
    assert 'pitch_names' not in note.to_dict(), 'the table must not bloat it'


def test_b_sharp_takes_the_octave_of_its_letter():
    """C# minor's leading tone at MIDI 60 is B#3, the pitch of C4. Counting
    the octave from the pitch would write B#4, an octave too high."""
    c_sharp_minor = KeyEstimate(tonic=1, is_major=False,
                                confidence=0.9).pitch_names
    assert [midi_to_name(m, c_sharp_minor) for m in (59, 60, 61, 72)] == \
        ['B3', 'B#3', 'C#4', 'B#4']


def test_transposing_a_note_keeps_its_pitch_names():
    note = TranscribedNote(61, 0.0, 0.5, 0.9, pitch_names=D_MINOR.pitch_names)
    moved = note.transposed(9)
    assert (moved.midi, moved.name) == (70, 'Bb4')
    assert moved.pitch_names is note.pitch_names
    assert (note.midi, note.name) == (61, 'C#4'), 'the original must not move'


# --------------------------------------------------------------------------
# Every output format
# --------------------------------------------------------------------------

def _d_minor_notes():
    """C#, Bb and D in D minor. Sharps alone would write A#, flats alone Db,
    so neither plain table can pass for the key's own. The Bb is uncertain,
    so the lead sheet brackets it."""
    names = D_MINOR.pitch_names
    return [TranscribedNote(61, 0.0, 0.5, 0.9, lyric='one', pitch_names=names),
            TranscribedNote(70, 0.5, 1.0, 0.3, lyric='two', pitch_names=names),
            TranscribedNote(62, 1.0, 1.5, 0.9, lyric='three',
                            pitch_names=names)]


def test_every_text_format_names_notes_the_same_way():
    notes = _d_minor_notes()
    names = [n.to_dict()['note'] for n in notes]
    assert names == ['C#4', 'Bb4', 'D4']

    rows = list(csv.DictReader(io.StringIO(format_csv(notes))))
    assert [r['note'] for r in rows] == names
    assert [int(r['midi']) for r in rows] == [61, 70, 62]

    payload = json.loads(format_json(notes))
    assert [n['note'] for n in payload['notes']] == names
    assert [n['midi'] for n in payload['notes']] == [61, 70, 62]

    for text in (format_table(notes), format_leadsheet(notes)):
        assert all(name in text for name in names)
        assert not any(wrong in text for wrong in ('Db4', 'A#4'))
    assert '(Bb4)' in format_leadsheet(notes)

    # LRC carries lyrics and times only, so there is nothing to spell.
    plain = [replace(n, pitch_names=SHARPS) for n in notes]
    assert format_lrc(notes) == format_lrc(plain)


def test_midi_export_is_untouched_by_spelling(tmp_path, monkeypatch):
    """Any table must hand pretty_midi identical notes. It is stood in for,
    so this runs whether or not it is installed."""
    handed = []

    class FakeMIDI:
        def __init__(self, initial_tempo):
            self.instruments = []

        def write(self, path):
            handed.append([vars(n) for i in self.instruments for n in i.notes])

    monkeypatch.setitem(sys.modules, 'pretty_midi', types.SimpleNamespace(
        PrettyMIDI=FakeMIDI,
        Instrument=lambda program: types.SimpleNamespace(notes=[]),
        Note=lambda **fields: types.SimpleNamespace(**fields)))

    notes = _d_minor_notes()
    write_midi(notes, tmp_path / 'd_minor.mid')
    write_midi([replace(n, pitch_names=SHARPS) for n in notes],
               tmp_path / 'sharps.mid')

    assert handed[0] == handed[1]
    assert [n['pitch'] for n in handed[0]] == [61, 70, 62]


# --------------------------------------------------------------------------
# Through the pipeline
# --------------------------------------------------------------------------

# Tonic, fourth and fifth of concert Gb major (Gb4, B4, Db5). Alto sax reads
# them as Eb, Ab and Bb - all three misnamed if spelled with sharps.
CONCERT_GB = (66, 71, 73)
GB_MAJOR = KeyEstimate(tonic=6, is_major=True, confidence=0.9)

# Tonic, leading tone and 6th of concert F minor (F4, E4, Db5). Alto sax
# reads them in D minor as D, C# and Bb: a sharp and a flat in one key.
CONCERT_F_MINOR = (65, 64, 73)
F_MINOR = KeyEstimate(tonic=5, is_major=False, confidence=0.9)


def _run(tmp_path, monkeypatch, key, transpose, concert=CONCERT_GB):
    """The real pipeline, with the key estimate and the sung notes supplied."""
    audio = tmp_path / 'vocals.wav'
    audio.write_bytes(b'\0' * 64)   # only has to exist: nothing decodes it

    def transcribe(self, audio_path, key_pitch_classes=None, progress=None):
        notes = [TranscribedNote(midi, i * 0.5, i * 0.5 + 0.4, 0.9)
                 for i, midi in enumerate(concert)]
        return TranscriptionResult(notes=notes, frames=None, duration=2.0,
                                   voters_used=['stand-in'])

    monkeypatch.setattr('meloscribe.pipeline.estimate_key', lambda path: key)
    monkeypatch.setattr(PitchEngine, 'transcribe', transcribe)
    return Pipeline().run(TranscriptionRequest(
        input_path=audio, lyrics_mode=LyricsMode.OFF, vocals_only=True,
        transpose=transpose))


def test_alto_part_of_a_gb_major_song_is_written_in_eb_with_flats(
        tmp_path, monkeypatch):
    output = _run(tmp_path, monkeypatch, GB_MAJOR, ALTO_SAX)

    assert [n.midi for n in output.notes] == [m + ALTO_SAX for m in CONCERT_GB]
    assert [n.name for n in output.notes] == ['Eb5', 'Ab5', 'Bb5']
    assert (output.spelling, output.pitch_names) == ('flat', SPELLINGS['flat'])

    summary = output.to_dict()
    assert (summary['key'], summary['written_key']) == ('Gb major', 'Eb major')
    assert [n['note'] for n in summary['notes']] == ['Eb5', 'Ab5', 'Bb5']


def test_alto_part_of_an_f_minor_song_sharpens_d_minors_leading_tone(
        tmp_path, monkeypatch):
    output = _run(tmp_path, monkeypatch, F_MINOR, ALTO_SAX, CONCERT_F_MINOR)

    assert [n.midi for n in output.notes] == [74, 73, 82]
    assert [n.name for n in output.notes] == ['D5', 'C#5', 'Bb5']
    assert (output.spelling, output.pitch_names) == ('flat', D_MINOR.pitch_names)
    assert output.to_dict()['written_key'] == 'D minor'


def test_untransposed_output_has_no_separate_written_key(tmp_path, monkeypatch):
    output = _run(tmp_path, monkeypatch, GB_MAJOR, 0)

    # Gb major's 4th is Cb - written an octave up from the B it sounds as.
    assert [n.name for n in output.notes] == ['Gb4', 'Cb5', 'Db5']
    assert output.written_key is None
    assert output.to_dict()['written_key'] is None


def test_uncertain_key_spells_the_same_pitches_with_sharps(tmp_path,
                                                           monkeypatch):
    unsure = KeyEstimate(tonic=6, is_major=True, confidence=0.2)
    trusted = _run(tmp_path, monkeypatch, GB_MAJOR, ALTO_SAX)
    output = _run(tmp_path, monkeypatch, unsure, ALTO_SAX)

    assert [n.midi for n in output.notes] == [n.midi for n in trusted.notes]
    assert [n.name for n in output.notes] == ['D#5', 'G#5', 'A#5']
    assert (output.spelling, output.pitch_names) == ('sharp', SHARPS)
    assert any('spelling notes with sharps' in w for w in output.warnings)


# --------------------------------------------------------------------------
# API and CLI
# --------------------------------------------------------------------------

def test_notes_payload_reports_the_written_key_and_its_pitch_names(
        tmp_path, monkeypatch):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient

    from meloscribe.api import app as server
    from meloscribe.api.jobs import JobStatus

    output = _run(tmp_path, monkeypatch, F_MINOR, ALTO_SAX, CONCERT_F_MINOR)
    job = server.jobs.create(filename='vocals.wav',
                             params={'transpose': ALTO_SAX})
    job.result, job.status = output, JobStatus.DONE
    client = TestClient(server.app)

    body = client.get(f"/api/jobs/{job.id}/notes").json()
    assert (body['key'], body['written_key'], body['spelling']) == \
        ('F minor', 'D minor', 'flat')
    assert body['pitch_names'] == list(D_MINOR.pitch_names)
    assert [n['note'] for n in body['notes']] == ['D5', 'C#5', 'Bb5']
    # What the piano roll does with the table must give each note's own name.
    assert all(n['note'] == midi_to_name(n['midi'], body['pitch_names'])
               for n in body['notes'])

    # A download renders the same notes, so it must name them the same way.
    rows = csv.DictReader(io.StringIO(
        client.get(f"/api/jobs/{job.id}/download/csv").text))
    assert [r['note'] for r in rows] == ['D5', 'C#5', 'Bb5']


def test_cli_names_the_written_key_beside_the_concert_key(tmp_path,
                                                          monkeypatch, capsys):
    output = _run(tmp_path, monkeypatch, GB_MAJOR, ALTO_SAX)
    monkeypatch.setattr(Pipeline, 'run',
                        lambda self, request, progress=None: output)

    assert cli.main([str(tmp_path / 'vocals.wav'), '--no-lyrics',
                     '--vocals-only', '--transpose', str(ALTO_SAX),
                     '--format', 'json']) == 0
    printed = capsys.readouterr()

    assert 'key: Gb major (confidence 0.90)  written: Eb major (+9)' in printed.err
    payload = json.loads(printed.out)
    assert (payload['key'], payload['written_key']) == ('Gb major', 'Eb major')
    assert [n['note'] for n in payload['notes']] == ['Eb5', 'Ab5', 'Bb5']
