"""Sheet-music export: quantisation and the MusicXML written from it.

The load-bearing properties: a performed rhythm comes back as the written one
(through timing jitter, legato releases and pickups), no note is ever dropped
to make the grid fit, every measure adds up exactly, and the file says what
the other outputs say - the same spelled pitches, the written key, and a
transposition a notation program can undo. Nothing here needs models,
network or audio.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from meloscribe import notation, rhythm
from meloscribe.key import KeyEstimate, note_names
from meloscribe.lyrics.align import LyricLine, LyricWord, TimedLyrics, \
    attach_to_notes
from meloscribe.musicxml import (MEDIA_TYPE, choose_clef, clock, pitch_parts,
                                 to_musicxml, transpose_interval)
from meloscribe.output import FORMATS, render
from meloscribe.pitch.engine import TranscribedNote

BPM = 100.0
PERIOD = 60.0 / BPM
T = notation.TICKS_PER_BEAT

# Quarters, eighths, sixteenths, a dotted eighth and a written rest: straight
# material with no swing, twelve notes so the release estimate applies.
STRAIGHT = [(60, 1.0), (62, 0.5), (64, 0.5), (65, 0.25), (67, 0.25),
            (69, 0.5), (67, 1.0), (None, 1.0), (65, 0.5), (64, 0.5),
            (62, 0.75), (60, 0.25), (59, 2.0)]


def perform(values, start_beat=4.0, legato=0.92, jitter=0.0, seed=0):
    """Notes played from a pattern of (midi or None, beats), plus what was
    written: (beat, length) per note, relative to the first note."""
    rng = np.random.default_rng(seed)
    notes, written, beat = [], [], start_beat
    for midi, length in values:
        if midi is not None:
            on = beat * PERIOD + rng.uniform(-jitter, jitter)
            end = (beat + length * legato) * PERIOD + rng.uniform(-jitter,
                                                                  jitter)
            notes.append(TranscribedNote(midi=midi, start=on, end=end,
                                         confidence=0.9))
            written.append((beat - start_beat, length))
        beat += length
    return notes, written


def grid(confidence=1.0):
    return rhythm.constant_tempo_grid(BPM, 200.0, confidence=confidence)


def read_back(score):
    return [(round(q.start / T, 4), round(q.duration / T, 4))
            for q in score.notes]


def exact(written):
    return [(round(b, 4), round(v, 4)) for b, v in written]


def parse(xml):
    return ET.fromstring(xml.encode('utf-8'))


# --------------------------------------------------------------------------
# Quantisation
# --------------------------------------------------------------------------

def test_straight_eighths_and_sixteenths_come_back_as_written():
    notes, written = perform(STRAIGHT)
    score = notation.quantize(notes, grid())
    assert read_back(score) == exact(written)
    assert score.ternary_beats == frozenset()
    assert not score.approximate


@pytest.mark.parametrize('seed', range(12))
def test_thirty_ms_of_jitter_changes_nothing(seed):
    """+/-30ms on every onset and release, as a transcription has - at 100
    BPM that is a twentieth of a beat, well inside a sixteenth, and must not
    tip a straight line into triplets either."""
    notes, written = perform(STRAIGHT, jitter=0.03, seed=seed)
    score = notation.quantize(notes, grid())
    assert read_back(score) == exact(written)
    assert score.ternary_beats == frozenset()


def test_triplets_get_a_triplet_beat():
    values = [(60, 1 / 3), (62, 1 / 3), (64, 1 / 3), (65, 1.0),
              (67, 1 / 3), (65, 1 / 3), (64, 1 / 3), (62, 1.0),
              (60, 1.0), (62, 0.5), (64, 0.5), (65, 2.0)]
    notes, written = perform(values)
    score = notation.quantize(notes, grid())
    assert read_back(score) == exact(written)
    assert score.ternary_beats == frozenset({0, 2})

    measures = score.measures()
    notation.check_layout(measures)
    triplets = [p for p in measures[0].pieces if p.triplet]
    assert len(triplets) == 6
    assert triplets[0].tuplet_start and triplets[2].tuplet_stop
    assert {p.type for p in triplets} == {'eighth'}


def test_a_note_across_the_bar_line_is_tied():
    notes, _ = perform([(60, 3.0), (62, 2.0), (64, 3.0)])
    measures = notation.quantize(notes, grid()).measures()
    notation.check_layout(measures)
    held = [p for m in measures for p in m.pieces if p.note is not None
            and p.note.midi == 62]
    assert [(p.type, p.tie_start, p.tie_stop) for p in held] == [
        ('quarter', True, False), ('quarter', False, True)]


def test_off_beat_notes_are_tied_over_the_beat():
    """A syncopated quarter is written eighth-tied-eighth, so the beat stays
    visible."""
    notes, _ = perform([(60, 0.5), (62, 1.0), (64, 0.5), (65, 2.0)])
    pieces = notation.quantize(notes, grid()).measures()[0].pieces
    assert [(p.duration, p.tie_start) for p in pieces
            if p.note is not None and p.note.midi == 62] == [(6, True),
                                                             (6, False)]


def test_pickup_puts_the_first_bar_line_after_it():
    notes, _ = perform([(67, 1.0), (72, 2.0), (71, 1.0), (69, 4.0)])
    score = notation.quantize(notes, grid(), pickup=1)
    measures = score.measures()
    notation.check_layout(measures)
    assert measures[0].implicit and measures[0].number == 0
    assert measures[0].length == T
    assert measures[1].number == 1 and measures[1].start == T

    plain = notation.quantize(notes, grid()).measures()
    assert plain[0].number == 1 and not plain[0].implicit


def test_three_four_bars():
    notes, _ = perform([(60, 3.0), (62, 1.0), (64, 2.0), (65, 3.0)])
    measures = notation.quantize(notes, grid(), beats_per_bar=3).measures()
    notation.check_layout(measures)
    assert {m.length for m in measures} == {3 * T}


def test_a_short_gap_is_absorbed_not_written_as_a_rest():
    notes = [TranscribedNote(60, 4 * PERIOD, 5 * PERIOD - 0.03, 0.9),
             TranscribedNote(62, 5 * PERIOD, 6 * PERIOD, 0.9)]
    score = notation.quantize(notes, grid())
    assert score.notes[0].end == score.notes[1].start
    assert not any(p.is_rest for p in score.measures()[0].pieces[:2])


def test_a_real_rest_survives():
    notes, written = perform([(60, 1.0), (None, 1.0), (62, 1.0)])
    score = notation.quantize(notes, grid())
    assert read_back(score) == exact(written)
    rests = [p for p in score.measures()[0].pieces if p.is_rest]
    assert rests[0].start == T and rests[0].duration == T


def test_no_note_is_dropped_to_fit_the_grid():
    """Two onsets 20ms apart would share a slot; each gets its own instead,
    at least the minimum value long."""
    notes = [TranscribedNote(60, 4 * PERIOD, 4 * PERIOD + 0.02, 0.4),
             TranscribedNote(62, 4 * PERIOD + 0.02, 5 * PERIOD, 0.9),
             TranscribedNote(64, 5 * PERIOD, 6 * PERIOD, 0.9)]
    score = notation.quantize(notes, grid())
    assert len(score.notes) == 3
    starts = [q.start for q in score.notes]
    assert starts == sorted(set(starts))
    assert min(q.duration for q in score.notes) >= 3


def test_an_unreliable_grid_falls_back_to_note_spacing():
    notes, written = perform([(60, 1.0)] * 8 + [(62, 2.0)])
    score = notation.quantize(notes, grid(confidence=0.0))
    assert score.approximate and score.grid_source == 'fallback'
    assert score.bpm == pytest.approx(BPM, rel=0.02)
    assert read_back(score) == exact(written)
    assert notation.quantize(notes, None).approximate


def test_regularising_removes_tracker_wobble():
    rng = np.random.default_rng(3)
    true = np.arange(40) * PERIOD + 1.0
    wobbly = rhythm.BeatGrid(bpm=BPM, beat_times=true + rng.normal(0, 0.04, 40),
                             confidence=0.9)
    smooth = notation.regularise(wobbly)
    assert np.std(smooth.beat_times - true) < 0.5 * np.std(
        wobbly.beat_times - true)


def test_quantising_never_touches_the_notes():
    notes, _ = perform(STRAIGHT, jitter=0.03)
    before = [(n.start, n.end, n.midi) for n in notes]
    notation.quantize(notes, grid())
    assert [(n.start, n.end, n.midi) for n in notes] == before


# --------------------------------------------------------------------------
# Lyrics
# --------------------------------------------------------------------------

def _worded(words_per_note):
    """One note per beat; `words_per_note[i]` is the index of the word note
    i is sung on."""
    notes = [TranscribedNote(60 + i, (4 + i) * PERIOD, (5 + i) * PERIOD - 0.05,
                             0.9) for i in range(len(words_per_note))]
    texts = ['la', 'la', 'love', 'me']
    words = []
    for k in sorted(set(words_per_note)):
        idx = [i for i, w in enumerate(words_per_note) if w == k]
        words.append(LyricWord(text=texts[k], start=notes[idx[0]].start,
                               end=notes[idx[-1]].end))
    lyrics = TimedLyrics(lines=[LyricLine(start=words[0].start, text='la la '
                                          'love me', words=words)],
                         words=words)
    attach_to_notes(notes, lyrics)
    return notes, lyrics


def test_a_word_goes_under_its_first_note_and_a_melisma_gets_none():
    notes, lyrics = _worded([0, 1, 2, 2, 2, 3])
    marks = notation.lyric_marks(notes, lyrics)
    assert [m.text if m else None for m in marks] == [
        'la', 'la', 'love', None, None, 'me']
    assert marks[2].extend and not marks[0].extend
    # 'la la' is two words: the note labels alone read it as one held word.
    assert [m.text if m else None for m in notation.lyric_marks(notes)] == [
        'la', None, 'love', None, None, 'me']


def test_line_lyrics_go_on_the_line_s_first_note():
    notes, _ = perform([(60, 1.0)] * 6)
    lines = [LyricLine(start=notes[0].start - 0.1, text='first line',
                       end=notes[2].end + 0.01),
             LyricLine(start=notes[3].start - 0.1, text='second line')]
    lyrics = TimedLyrics(lines=lines)
    attach_to_notes(notes, lyrics)
    marks = notation.lyric_marks(notes, lyrics)
    assert [(m.text, m.kind) if m else None for m in marks] == [
        ('first line', 'line'), None, None, ('second line', 'line'), None,
        None]


# --------------------------------------------------------------------------
# MusicXML
# --------------------------------------------------------------------------

GB_MAJOR = KeyEstimate(tonic=6, is_major=True, confidence=0.9)


def sheet(values=STRAIGHT, key=GB_MAJOR, transpose=0, pickup=0, **kwargs):
    notes, _ = perform(values, jitter=0.02)
    names = note_names(key, transpose)
    notes = [n.transposed(transpose) for n in notes]
    for note in notes:
        note.pitch_names = names
    score = notation.quantize(notes, grid(), pickup=pickup)
    return parse(to_musicxml(score, key=key, transpose=transpose, **kwargs))


def test_every_measure_adds_up_exactly():
    root = sheet(STRAIGHT + [(67, 1 / 3), (65, 1 / 3), (64, 1 / 3), (62, 3.5)],
                 pickup=1)
    divisions = int(root.find('.//divisions').text)
    beats = int(root.find('.//time/beats').text)
    measures = root.findall('part/measure')
    assert measures[0].get('implicit') == 'yes'
    for measure in measures:
        total = sum(int(n.find('duration').text)
                    for n in measure.findall('note'))
        expected = divisions if measure.get('implicit') else divisions * beats
        assert total == expected, measure.get('number')


def test_key_and_transposition_for_alto_sax():
    """Concert Gb major, raised 9 for alto sax: written in Eb major (three
    flats), and <transpose> takes it back down a major sixth."""
    root = sheet(transpose=9)
    assert root.find('.//key/fifths').text == '-3'
    assert root.find('.//key/mode').text == 'major'
    assert root.find('.//transpose/diatonic').text == '-5'
    assert root.find('.//transpose/chromatic').text == '-9'
    assert root.find('.//transpose/octave-change') is None
    first = root.find('.//note/pitch')                   # C4 + 9 = A4
    assert (first.find('step').text, first.find('alter'),
            first.find('octave').text) == ('A', None, '4')
    third = list(root.iter('pitch'))[2]                   # E4 + 9 = Db5
    assert [third.find(t).text for t in ('step', 'alter', 'octave')] == [
        'D', '-1', '5']

    concert = sheet()
    assert concert.find('.//key/fifths').text == '-6'
    assert concert.find('.//transpose') is None


def test_an_untrusted_key_writes_no_signature():
    root = sheet(key=KeyEstimate(tonic=3, is_major=True, confidence=0.2))
    assert root.find('.//key/fifths').text == '0'
    assert root.find('.//key/mode') is None


@pytest.mark.parametrize('transpose,expected', [
    (9, (-5, -9, 0)),      # alto sax: a major sixth
    (2, (-1, -2, 0)),      # Bb trumpet: a major second
    (14, (-1, -2, -1)),    # tenor sax: a major ninth
    (12, (0, 0, -1)),      # an octave
    (-3, (2, 3, 0)),       # written a minor third below
])
def test_transpose_interval(transpose, expected):
    assert transpose_interval(transpose, None) == expected


def test_transpose_interval_follows_the_key_spelling():
    """Concert F# minor for alto sax is written Eb minor (key.py's spelling),
    so the way back is a diminished seventh, not a major sixth - which would
    give Gb minor, a key with nine flats."""
    f_sharp_minor = KeyEstimate(tonic=6, is_major=False, confidence=0.9)
    assert transpose_interval(9, f_sharp_minor) == (-6, -9, 0)


@pytest.mark.parametrize('name,parts', [
    ('Eb4', ('E', -1, 4)), ('B#3', ('B', 1, 3)), ('Cb5', ('C', -1, 5)),
    ('E#4', ('E', 1, 4)), ('F##4', ('F', 2, 4)), ('C-1', ('C', 0, -1)),
])
def test_pitch_parts_keep_the_letter_s_octave(name, parts):
    assert pitch_parts(name) == parts


def test_b_sharp_is_written_below_c():
    """C# minor's leading tone at MIDI 60 is B#3, the pitch of C4: the file
    must say step B, alter 1, octave 3 - not octave 4."""
    c_sharp_minor = KeyEstimate(tonic=1, is_major=False, confidence=0.9)
    root = sheet([(60, 1.0), (61, 3.0)], key=c_sharp_minor)
    pitch = root.find('.//note/pitch')
    assert [pitch.find(t).text for t in ('step', 'alter', 'octave')] == [
        'B', '1', '3']


@pytest.mark.parametrize('names,clef', [
    (['C4', 'E4', 'G5'], 'treble'),
    (['C4', 'D4', 'E4', 'G4', 'A4', 'C4'], 'treble'),   # middle C is fine
    (['C3', 'E3', 'G3', 'C4', 'E4'], 'treble-8vb'),
    (['E2', 'G2', 'C3', 'E3'], 'bass'),
])
def test_clef_by_range(names, clef):
    assert choose_clef(names) == clef


def test_low_voice_gets_an_octave_treble_clef():
    root = sheet([(52, 1.0), (55, 1.0), (60, 1.0), (64, 2.0)])   # E3-E4
    assert root.find('.//clef/sign').text == 'G'
    assert root.find('.//clef/clef-octave-change').text == '-1'


def test_tempo_title_and_artist():
    root = sheet(title='Night Song', artist='Someone')
    assert root.find('work/work-title').text == 'Night Song'
    assert root.find('identification/creator').text == 'Someone'
    assert root.find('.//sound').get('tempo') == '100'
    assert root.find('.//metronome/per-minute').text == '100'
    credits = {c.find('credit-type').text: c.find('credit-words').text
               for c in root.findall('credit')}
    assert credits == {'title': 'Night Song', 'composer': 'Someone'}


def test_uncertain_notes_are_coloured_and_approximate_rhythm_is_said():
    notes, _ = perform([(60, 1.0)] * 6)
    notes[2].confidence = 0.2
    score = notation.quantize(notes, None)
    root = parse(to_musicxml(score))
    coloured = [n for n in root.iter('note') if n.get('color')]
    assert len(coloured) == 1
    assert root.find('.//miscellaneous-field').text == 'approximate'
    assert any('Approximate rhythm' in (w.text or '')
               for w in root.iter('words'))


def test_word_lyrics_in_the_file():
    notes, lyrics = _worded([0, 1, 2, 2, 2, 3])
    root = parse(to_musicxml(notation.quantize(notes, grid(), lyrics=lyrics)))
    sung = [n.find('lyric') for n in root.iter('note')
            if n.find('pitch') is not None]
    assert [l.find('text').text if l is not None else None for l in sung] == [
        'la', 'la', 'love', None, None, 'me']
    assert sung[2].find('extend') is not None
    assert {l.find('syllabic').text for l in sung if l is not None} == {
        'single'}


def test_render_offers_musicxml():
    notes, _ = perform(STRAIGHT)
    assert 'musicxml' in FORMATS
    root = parse(render(notes, 'musicxml', grid=grid(), title='T'))
    assert root.tag == 'score-partwise' and root.get('version') == '4.0'
    assert MEDIA_TYPE == 'application/vnd.recordare.musicxml+xml'


# --------------------------------------------------------------------------
# The key signature and the libraries' own conversions
# --------------------------------------------------------------------------

@pytest.mark.parametrize('tonic', range(12))
@pytest.mark.parametrize('is_major', [True, False])
def test_fifths_agree_with_pretty_midi(tonic, is_major):
    pretty_midi = pytest.importorskip('pretty_midi')
    key = KeyEstimate(tonic=tonic, is_major=is_major, confidence=0.9)
    _, accidentals = pretty_midi.key_number_to_mode_accidentals(
        tonic + (0 if is_major else 12))
    # pretty_midi picks F# major for pitch class 6; key.py writes Gb - the
    # same place on the circle, so equal modulo 12.
    assert (key.fifths - accidentals) % 12 == 0
    assert (key.fifths < 0) == (key.spelling == 'flat')


@pytest.mark.parametrize('tonic', range(12))
@pytest.mark.parametrize('is_major', [True, False])
def test_fifths_agree_with_music21(tonic, is_major):
    music21 = pytest.importorskip('music21')
    key = KeyEstimate(tonic=tonic, is_major=is_major, confidence=0.9)
    name = key.name.split()[0].replace('b', '-')
    assert music21.key.Key(name if is_major else name.lower()).sharps == \
        key.fifths


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------

def test_music21_reads_back_what_was_written():
    music21 = pytest.importorskip('music21')
    values = [(66, 1.0), (68, 0.5), (70, 0.5), (71, 1 / 3), (70, 1 / 3),
              (68, 1 / 3), (66, 3.0), (None, 1.0), (63, 2.5), (61, 1.5)]
    notes, written = perform(values, jitter=0.02)
    names = note_names(GB_MAJOR, 9)
    notes = [n.transposed(9) for n in notes]
    for note in notes:
        note.pitch_names = names
    xml = to_musicxml(notation.quantize(notes, grid(), pickup=1),
                      key=GB_MAJOR, transpose=9)
    parsed = music21.converter.parseData(xml, format='musicxml').stripTies()
    got = [n for n in parsed.recurse().notes]
    assert [n.pitch.nameWithOctave.replace('-', 'b') for n in got] == [
        n.name for n in notes]
    assert [float(n.duration.quarterLength) for n in got] == pytest.approx(
        [length for _, length in written])
    part = parsed.parts[0]
    assert part.getInstrument().transposition.semitones == -9
    sounding = [n.pitch.midi for n in parsed.toSoundingPitch().recurse().notes]
    assert sounding == [midi for midi, _ in values if midi is not None]


@pytest.mark.parametrize('seconds,shown', [
    (1.0, '0:01.0'), (61.04, '1:01.0'), (59.95, '1:00.0'), (119.97, '2:00.0'),
    (3599.99, '60:00.0')])
def test_the_start_time_never_reads_sixty_seconds(seconds, shown):
    # Split into minutes before rounding, 119.97 s read '1:60.0'.
    assert clock(seconds) == shown


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def test_cli_writes_sheet_music_and_quantised_midi(tmp_path, monkeypatch):
    pretty_midi = pytest.importorskip('pretty_midi')
    from meloscribe import cli
    from meloscribe.pipeline import TranscriptionOutput

    class FakePipeline:
        def run(self, request, progress=None):
            notes, _ = perform(STRAIGHT)
            names = note_names(GB_MAJOR, request.transpose)
            notes = [n.transposed(request.transpose) for n in notes]
            for note in notes:
                note.pitch_names = names
            return TranscriptionOutput(notes=notes, key=GB_MAJOR,
                                       request=request, pitch_names=names)

    monkeypatch.setattr(cli, 'Pipeline', FakePipeline)
    sheet_path, midi_path = tmp_path / 'sax.musicxml', tmp_path / 'sax.mid'
    printed = tmp_path / 'printed.musicxml'
    # No such audio: beat tracking fails, and the export falls back.
    assert cli.main([str(tmp_path / 'missing.mp3'), '--no-lyrics', '--quiet',
                     '--transpose', '9', '--pickup', '1',
                     '--musicxml', str(sheet_path), '--midi', str(midi_path),
                     '--quantize', '--format', 'musicxml',
                     '-o', str(printed)]) == 0

    root = ET.parse(str(sheet_path)).getroot()
    assert root.find('.//transpose/chromatic').text == '-9'
    assert root.find('part/measure').get('implicit') == 'yes'
    assert printed.read_text(encoding='utf-8') == \
        sheet_path.read_text(encoding='utf-8')

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    beat = 60.0 / midi.get_tempo_changes()[1][0]
    positions = [n.start / beat * T for n in midi.instruments[0].notes]
    assert positions and all(abs(p - round(p)) < 1e-3 for p in positions)
    assert midi.time_signature_changes[0].numerator == 4


# The CLI in its own process, so its stdout is a real redirected stream.
_REDIRECTED_CLI = '''
import sys
from meloscribe import cli
from meloscribe.key import KeyEstimate, note_names
from meloscribe.pipeline import TranscriptionOutput
from meloscribe.pitch.engine import TranscribedNote

KEY = KeyEstimate(tonic=0, is_major=True, confidence=0.9)


class FakePipeline:
    def run(self, request, progress=None):
        names = note_names(KEY, request.transpose)
        notes = []
        for i, midi in enumerate((60, 62, 64, 65)):
            note = TranscribedNote(midi, 1.0 + 0.5 * i, 1.4 + 0.5 * i, 0.9,
                                   lyric=sys.argv[2], syllable=sys.argv[2])
            note.pitch_names = names
            notes.append(note)
        return TranscriptionOutput(notes=notes, key=KEY, request=request,
                                   pitch_names=names)


cli.Pipeline = FakePipeline
sys.exit(cli.main(['missing.mp3', '--no-lyrics', '--quiet',
                   '--format', sys.argv[1]]))
'''


@pytest.mark.parametrize('pickup,meter', [('-1', '4/4'), ('4', '4/4'),
                                           ('3', '3/4')])
def test_a_pickup_outside_the_bar_is_refused(tmp_path, pickup, meter, capsys):
    from meloscribe import cli
    with pytest.raises(SystemExit) as stop:
        cli.main([str(tmp_path / 'song.mp3'), '--no-lyrics', '--musicxml',
                  str(tmp_path / 'out.musicxml'), '--pickup', pickup,
                  '--time-signature', meter])
    assert stop.value.code == 2
    assert '--pickup must be 0 to' in capsys.readouterr().err


@pytest.mark.parametrize('fmt', ['musicxml', 'json', 'csv', 'lrc'])
def test_redirected_output_is_utf8_whatever_the_console(tmp_path, fmt):
    """Redirected on Windows, stdout is in the ANSI code page: cp1252 wrote a
    curly apostrophe as a byte no UTF-8 reader accepts - in MusicXML that
    says it is UTF-8 - and could not encode a '♪' at all."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    lyric = 'don’t ♪ 愛'
    script = tmp_path / 'redirected_cli.py'
    script.write_text(_REDIRECTED_CLI, encoding='utf-8')
    env = {k: v for k, v in os.environ.items()
           if k not in ('PYTHONUTF8', 'PYTHONIOENCODING')}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
    done = subprocess.run([sys.executable, str(script), fmt, lyric],
                          capture_output=True, cwd=tmp_path, env=env,
                          timeout=300)
    assert done.returncode == 0, done.stderr.decode('utf-8', 'replace')[-2000:]
    text = done.stdout.decode('utf-8')      # not UTF-8: UnicodeDecodeError
    if fmt == 'musicxml':
        root = ET.fromstring(done.stdout)
        assert lyric in [t.text for t in root.iter('text')]
    elif fmt == 'json':
        import json
        assert json.loads(text)['notes'][0]['lyric'] == lyric
    else:
        assert lyric in text
