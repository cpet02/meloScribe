"""MusicXML 4.0 export: a quantised score as sheet music.

Written directly with ElementTree rather than through a notation library: the
format needs a few dozen element types, and a runtime dependency the size of
music21 for that would outweigh the feature. music21 is used only by the tests,
to prove that what is written here reads back as the music that was meant.

What the file carries, and where each decision comes from:

- Pitches are the notes' own spelled names (`TranscribedNote.name`), split
  into step, alter and octave. The octave is the letter's, as `midi_to_name`
  counts it, so B#3 stays below C4 and never jumps an octave.
- The key signature is the written key, and only when the notes are spelled
  from it (`key.signature_key`): a signature must never contradict the names
  under it.
- A transposed score carries `<transpose>` (written to sounding), so MuseScore
  and friends can flip it to concert pitch. The interval is spelled from the
  concert and written keys, so both views agree with `pitch_names`.
- The clef is whichever of treble, treble-8vb (the usual clef for a low male
  voice) and bass needs the fewest ledger lines.
- Uncertain notes are coloured, as every human-readable format here marks
  them: a reader should know which notes to check by ear.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from .key import KeyEstimate, signature_key
from .notation import LyricMark, Piece, QuantizedScore, TICKS_PER_BEAT

MEDIA_TYPE = 'application/vnd.recordare.musicxml+xml'
EXTENSION = 'musicxml'

DOCTYPE = ('<!DOCTYPE score-partwise PUBLIC '
           '"-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">')

# Below this a note is coloured, matching `output.LOW_CONFIDENCE` - the same
# line the table's '??' and the lead sheet's brackets are drawn at.
LOW_CONFIDENCE = 0.5
UNCERTAIN_COLOR = '#D55E00'

_NAME = re.compile(r'^([A-G])(#{1,2}|b{1,2})?(-?\d+)$')
_LETTERS = 'CDEFGAB'

# Clef: (sign, line, octave change, bottom line, top line) - the staff's
# outer lines as diatonic steps (octave * 7 + letter), for counting ledger
# lines. Treble-8vb reads a written note an octave above where it sounds.
CLEFS: Dict[str, Tuple[str, int, int, int, int]] = {
    'treble': ('G', 2, 0, 4 * 7 + 2, 5 * 7 + 3),        # E4 .. F5
    'treble-8vb': ('G', 2, -1, 3 * 7 + 2, 4 * 7 + 3),   # E3 .. F4
    'bass': ('F', 4, 0, 2 * 7 + 4, 3 * 7 + 5),          # G2 .. A3
}


def pitch_parts(name: str) -> Tuple[str, int, int]:
    """'Eb4' -> ('E', -1, 4); 'B#3' -> ('B', 1, 3); 'Cb5' -> ('C', -1, 5).

    The octave is taken as written, because `midi_to_name` already counts it
    by the letter: B#3 sounds as C4 and Cb5 as B4, and recomputing the octave
    from the pitch would move both by an octave."""
    match = _NAME.match(name)
    if not match:
        raise ValueError(f"not a spelled note name: {name!r}")
    letter, accidentals, octave = match.groups()
    accidentals = accidentals or ''
    return letter, accidentals.count('#') - accidentals.count('b'), int(octave)


def _staff_step(name: str) -> int:
    letter, _, octave = pitch_parts(name)
    return octave * 7 + _LETTERS.index(letter)


def choose_clef(names: Sequence[str]) -> str:
    """The clef that puts these notes on the staff with the fewest ledger
    lines. The first ledger line is free - middle C in the treble clef is
    ordinary reading, and charging for it put a C4-A4 tune in treble-8vb.
    Ties go to treble, then treble-8vb: they are what most readers of a
    melody expect."""
    if not names:
        return 'treble'
    steps = [_staff_step(n) for n in names]

    def ledger_lines(clef: str) -> int:
        _, _, _, bottom, top = CLEFS[clef]
        return sum(max(0, max(0, bottom - s) // 2 + max(0, s - top) // 2 - 1)
                   for s in steps)

    return min(CLEFS, key=lambda clef: (ledger_lines(clef),
                                        list(CLEFS).index(clef)))


def transpose_interval(transpose: int,
                       key: Optional[KeyEstimate]) -> Tuple[int, int, int]:
    """(diatonic, chromatic, octave-change) from written to sounding pitch.

    MusicXML's <transpose> says what to add to the written note to get the
    sounding one - the reverse of `transpose`, which raises the sung notes to
    the written ones. The diatonic step count is the distance between the
    letters of the concert and the written key's tonics, as `key.py` spells
    them, taken the way the semitones go: so the concert view MuseScore
    derives from it is spelled in the same key the notes were named from.
    Without a trusted key, C major stands in, which gives the conventional
    interval (a major sixth for alto sax).
    """
    concert = key if key is not None and key.as_prior() is not None \
        else KeyEstimate(tonic=0, is_major=True, confidence=1.0)
    written = concert.transposed(transpose)
    chromatic = -int(transpose)
    octaves = int(chromatic / 12)             # toward zero, keeps the sign
    chromatic -= 12 * octaves
    steps = (_LETTERS.index(_tonic_letter(concert))
             - _LETTERS.index(_tonic_letter(written))) % 7
    # Of the step counts that name this letter, the one nearest the size of
    # the interval: 9 semitones down is 5 steps down, never 2 up.
    target = chromatic * 7 / 12
    diatonic = min((steps + 7 * k for k in (-2, -1, 0, 1)),
                   key=lambda d: abs(d - target))
    return diatonic, chromatic, octaves


def _tonic_letter(key: KeyEstimate) -> str:
    return key.name[0]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def to_musicxml(score: QuantizedScore, *, key: Optional[KeyEstimate] = None,
                transpose: int = 0, title: str = '', artist: str = '',
                part_name: str = 'Melody', program: int = 65,
                software: str = 'meloScribe') -> str:
    """Render a quantised score as a MusicXML 4.0 partwise document.

    `key` is the concert key (`TranscriptionOutput.key`) and `transpose` the
    semitones the notes were raised by; the notes themselves are already the
    written ones. `program` is the General MIDI program, 0-based as in
    `output.write_midi`, whose alto-sax default it shares.
    """
    measures = score.measures()
    written_key = signature_key(key, transpose)

    root = ET.Element('score-partwise', version='4.0')
    if title:
        work = ET.SubElement(root, 'work')
        ET.SubElement(work, 'work-title').text = title
        ET.SubElement(root, 'movement-title').text = title

    ident = ET.SubElement(root, 'identification')
    if artist:
        # MusicXML has no 'performer' creator; 'composer' is where notation
        # programs look for the name under the title.
        ET.SubElement(ident, 'creator', type='composer').text = artist
    encoding = ET.SubElement(ident, 'encoding')
    ET.SubElement(encoding, 'software').text = software
    ET.SubElement(encoding, 'encoding-date').text = date.today().isoformat()
    # No beams, stems or accidentals are written: each program draws its own
    # from the durations, pitches and key, which is what they do best.
    for element in ('accidental', 'beam', 'stem'):
        ET.SubElement(encoding, 'supports', element=element, type='no')
    misc = ET.SubElement(ident, 'miscellaneous')
    ET.SubElement(misc, 'miscellaneous-field', name='meloscribe-rhythm').text = \
        'approximate' if score.approximate else score.grid_source

    for text, kind, justify in ((title, 'title', 'center'),
                                (artist, 'composer', 'right')):
        if text:
            credit = ET.SubElement(root, 'credit', page='1')
            ET.SubElement(credit, 'credit-type').text = kind
            ET.SubElement(credit, 'credit-words', justify=justify,
                          valign='top').text = text

    part_list = ET.SubElement(root, 'part-list')
    score_part = ET.SubElement(part_list, 'score-part', id='P1')
    ET.SubElement(score_part, 'part-name').text = part_name
    instrument = ET.SubElement(score_part, 'score-instrument', id='P1-I1')
    ET.SubElement(instrument, 'instrument-name').text = part_name
    midi = ET.SubElement(score_part, 'midi-instrument', id='P1-I1')
    ET.SubElement(midi, 'midi-channel').text = '1'
    ET.SubElement(midi, 'midi-program').text = str(int(program) + 1)

    part = ET.SubElement(root, 'part', id='P1')
    clef = choose_clef([n.name for n in score.notes])
    for i, measure in enumerate(measures):
        element = ET.SubElement(part, 'measure', number=str(measure.number))
        if measure.implicit:
            element.set('implicit', 'yes')
        if i == 0:
            _first_attributes(element, score, written_key, key, transpose, clef)
            _opening_directions(element, score)
        for piece in measure.pieces:
            _write_piece(element, piece)

    ET.indent(root, space='  ')   # for anyone who opens it in an editor
    return ('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            + DOCTYPE + '\n' + ET.tostring(root, encoding='unicode') + '\n')


def _first_attributes(measure: ET.Element, score: QuantizedScore,
                      written_key: Optional[KeyEstimate],
                      concert_key: Optional[KeyEstimate], transpose: int,
                      clef: str) -> None:
    attributes = ET.SubElement(measure, 'attributes')
    ET.SubElement(attributes, 'divisions').text = str(TICKS_PER_BEAT)

    key = ET.SubElement(attributes, 'key')
    if written_key is not None:
        ET.SubElement(key, 'fifths').text = str(written_key.fifths)
        ET.SubElement(key, 'mode').text = \
            'major' if written_key.is_major else 'minor'
    else:
        # No trusted key: no signature, and the notes carry their own sharps.
        ET.SubElement(key, 'fifths').text = '0'

    time = ET.SubElement(attributes, 'time')
    ET.SubElement(time, 'beats').text = str(score.beats_per_bar)
    ET.SubElement(time, 'beat-type').text = '4'

    sign, line, octave_change, _, _ = CLEFS[clef]
    clef_el = ET.SubElement(attributes, 'clef')
    ET.SubElement(clef_el, 'sign').text = sign
    ET.SubElement(clef_el, 'line').text = str(line)
    if octave_change:
        ET.SubElement(clef_el, 'clef-octave-change').text = str(octave_change)

    if transpose:
        diatonic, chromatic, octaves = transpose_interval(transpose,
                                                          concert_key)
        element = ET.SubElement(attributes, 'transpose')
        ET.SubElement(element, 'diatonic').text = str(diatonic)
        ET.SubElement(element, 'chromatic').text = str(chromatic)
        if octaves:
            ET.SubElement(element, 'octave-change').text = str(octaves)


def _opening_directions(measure: ET.Element, score: QuantizedScore) -> None:
    bpm = int(round(score.bpm))
    direction = ET.SubElement(measure, 'direction', placement='above')
    kind = ET.SubElement(direction, 'direction-type')
    metronome = ET.SubElement(kind, 'metronome',
                              parentheses='yes' if score.approximate else 'no')
    ET.SubElement(metronome, 'beat-unit').text = 'quarter'
    ET.SubElement(metronome, 'per-minute').text = str(bpm)
    ET.SubElement(direction, 'sound', tempo=str(bpm))

    notes = []
    if score.approximate:
        notes.append('Approximate rhythm: no reliable beat was found, so the '
                     'tempo is estimated from the note spacing')
    if score.origin_s >= 1.0:
        minutes, seconds = divmod(score.origin_s, 60.0)
        notes.append(f"Starts {int(minutes)}:{seconds:04.1f} into the "
                     f"recording")
    for text in notes:
        _words(measure, text, placement='above')


def _words(measure: ET.Element, text: str, placement: str,
           italic: bool = False) -> None:
    direction = ET.SubElement(measure, 'direction', placement=placement)
    kind = ET.SubElement(direction, 'direction-type')
    words = ET.SubElement(kind, 'words')
    if italic:
        words.set('font-style', 'italic')
    words.text = text


def _write_piece(measure: ET.Element, piece: Piece) -> None:
    source = piece.note
    mark: Optional[LyricMark] = source.lyric if source is not None else None
    if piece.first_of_note and mark is not None and mark.kind == 'line':
        # A whole line as one syllable would stretch its note across the
        # page; as text under the staff it marks where the line starts.
        _words(measure, mark.text, placement='below', italic=True)

    note = ET.SubElement(measure, 'note')
    if source is not None and source.confidence < LOW_CONFIDENCE:
        note.set('color', UNCERTAIN_COLOR)

    if source is None:
        rest = ET.SubElement(note, 'rest')
        if piece.measure_rest:
            rest.set('measure', 'yes')
    else:
        step, alter, octave = pitch_parts(source.name)
        pitch = ET.SubElement(note, 'pitch')
        ET.SubElement(pitch, 'step').text = step
        if alter:
            ET.SubElement(pitch, 'alter').text = str(alter)
        ET.SubElement(pitch, 'octave').text = str(octave)

    ET.SubElement(note, 'duration').text = str(piece.duration)
    if piece.tie_stop:
        ET.SubElement(note, 'tie', type='stop')
    if piece.tie_start:
        ET.SubElement(note, 'tie', type='start')
    ET.SubElement(note, 'voice').text = '1'
    if not piece.measure_rest:
        ET.SubElement(note, 'type').text = piece.type
        for _ in range(piece.dots):
            ET.SubElement(note, 'dot')
    if piece.triplet:
        modification = ET.SubElement(note, 'time-modification')
        ET.SubElement(modification, 'actual-notes').text = '3'
        ET.SubElement(modification, 'normal-notes').text = '2'

    notations: List[ET.Element] = []
    if piece.tie_stop:
        notations.append(ET.Element('tied', type='stop'))
    if piece.tie_start:
        notations.append(ET.Element('tied', type='start'))
    if piece.tuplet_start:
        notations.append(ET.Element('tuplet', type='start', bracket='yes'))
    if piece.tuplet_stop:
        notations.append(ET.Element('tuplet', type='stop'))
    if notations:
        ET.SubElement(note, 'notations').extend(notations)

    if piece.first_of_note and mark is not None and mark.kind == 'word':
        lyric = ET.SubElement(note, 'lyric', number='1')
        ET.SubElement(lyric, 'syllabic').text = 'single'
        ET.SubElement(lyric, 'text').text = mark.text
        if mark.extend:
            ET.SubElement(lyric, 'extend')
