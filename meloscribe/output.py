"""Rendering a transcription into something usable.

Formats differ in what they are *for*, which is why several exist rather than
one canonical dump: `table` to read at a glance, `leadsheet` to play from,
`csv`/`json` to post-process, `lrc` to feed back into a player, `midi` to open
in a DAW, and `musicxml` to open as sheet music. Only `musicxml` (and MIDI
with `quantize`) moves notes onto a grid - see `notation.py` - because only
notation needs note values; every other format keeps the performed timing.

Confidence is surfaced in every human-readable format. A transcription is an
estimate, and a note the system is unsure about should look different from one
it is certain of - otherwise the user has no way to know where to check.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

from .pitch.engine import TranscribedNote

if TYPE_CHECKING:
    from .notation import QuantizedScore
    from .pipeline import TranscriptionOutput
    from .rhythm import BeatGrid

# Below this, a note is worth a second listen. Chosen to match the engine's
# own `low_confidence` default so the two never disagree.
LOW_CONFIDENCE = 0.5


def _confidence_mark(confidence: float) -> str:
    if confidence >= 0.8:
        return ' '
    if confidence >= LOW_CONFIDENCE:
        return '?'
    return '??'


def _rhythm_cells(note: TranscribedNote) -> List[str]:
    """The 'beat' and 'len' cells: where the note sat against the beat grid.

    Deviation is shown in beats rather than seconds so it stays comparable
    across tempi, and signed so the reader can tell a note that anticipates the
    beat from one that drags behind it - a systematic sign is itself
    informative, and rounding it away to a distance would hide that.
    """
    if note.beat_deviation is None:
        return ['', '']
    return [f"{note.beat_deviation:+.2f}", f"{note.duration_beats or 0.0:.2f}"]


def format_table(notes: Sequence[TranscribedNote],
                 show_voters: bool = False) -> str:
    """Fixed-width table, one row per note."""
    if not notes:
        return '(no notes detected)'

    # The rhythm columns appear only when the stage actually ran and found a
    # grid it trusted. Printing an empty 'beat' column on every free-time
    # recording would train the reader to ignore it.
    show_rhythm = any(n.beat_deviation is not None for n in notes)

    headers = ['#', 'note', 'start', 'end', 'dur', 'conf', '', 'cents']
    if show_rhythm:
        headers.extend(['beat', 'len'])
    headers.append('lyric')
    if show_voters:
        headers.extend(sorted(notes[0].voter_scores))

    rows: List[List[str]] = []
    for i, note in enumerate(notes, 1):
        row = [str(i), note.name, f"{note.start:.2f}", f"{note.end:.2f}",
               f"{note.duration:.2f}", f"{note.confidence:.2f}",
               _confidence_mark(note.confidence),
               f"{note.pitch_cents:+.0f}"]
        if show_rhythm:
            row.extend(_rhythm_cells(note))
        row.append((note.lyric or '')[:24])
        if show_voters:
            row.extend(f"{note.voter_scores.get(name, 0.0):.2f}"
                       for name in sorted(notes[0].voter_scores))
        rows.append(row)

    widths = [max(len(headers[i]), max(len(r[i]) for r in rows))
              for i in range(len(headers))]
    line = '  '.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, '-' * len(line)]
    out.extend('  '.join(c.ljust(widths[i]) for i, c in enumerate(row))
               for row in rows)
    return '\n'.join(out)


def format_csv(notes: Sequence[TranscribedNote]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['index', 'note', 'midi', 'start', 'end', 'duration',
                     'confidence', 'cents_off', 'beat_deviation',
                     'duration_beats', 'lyric'])
    for i, note in enumerate(notes, 1):
        writer.writerow([i, note.name, note.midi, f"{note.start:.3f}",
                         f"{note.end:.3f}", f"{note.duration:.3f}",
                         f"{note.confidence:.4f}", f"{note.pitch_cents:.1f}",
                         '' if note.beat_deviation is None
                         else f"{note.beat_deviation:.3f}",
                         '' if note.duration_beats is None
                         else f"{note.duration_beats:.3f}",
                         note.lyric or ''])
    return buffer.getvalue()


def format_json(notes: Sequence[TranscribedNote], **extra) -> str:
    return json.dumps({'notes': [n.to_dict() for n in notes], **extra}, indent=2)


def format_leadsheet(notes: Sequence[TranscribedNote],
                     line_width: int = 8) -> str:
    """Notes grouped into readable lines, with lyrics underneath.

    Laid out lyric-under-note like a real lead sheet, so a singer or player can
    follow it. Uncertain notes are bracketed rather than hidden - the reader
    decides what to trust, which they can only do if they are told.
    """
    if not notes:
        return '(no notes detected)'

    out: List[str] = []
    for start in range(0, len(notes), line_width):
        chunk = notes[start:start + line_width]

        cells = []
        for note in chunk:
            label = note.name
            if note.confidence < LOW_CONFIDENCE:
                label = f"({label})"
            cells.append(label)

        lyrics = [(note.lyric or '')[:7] for note in chunk]
        width = max(max(len(c) for c in cells), max(len(l) for l in lyrics)) + 2

        out.append(''.join(c.ljust(width) for c in cells))
        if any(lyrics):
            out.append(''.join(l.ljust(width) for l in lyrics))
        out.append('')

    return '\n'.join(out).rstrip()


def format_lrc(notes: Sequence[TranscribedNote], title: str = '',
               artist: str = '') -> str:
    """Export the aligned lyrics as an LRC file.

    Worth saving even when the lyrics came from LRClib: what comes back out has
    been refined against the vocal stem, so it is better timed than what went
    in, and can be reused directly.
    """
    out: List[str] = []
    if title:
        out.append(f"[ti:{title}]")
    if artist:
        out.append(f"[ar:{artist}]")

    previous: Optional[str] = None
    for note in notes:
        if not note.lyric or note.lyric == previous:
            continue  # one timestamp per lyric, not per note of a melisma
        minutes, seconds = divmod(max(0.0, note.start), 60)
        out.append(f"[{int(minutes):02d}:{seconds:05.2f}]{note.lyric}")
        previous = note.lyric

    return '\n'.join(out)


def write_midi(notes: Sequence[TranscribedNote], path,
               tempo: float = 120.0, program: int = 65,
               score: Optional['QuantizedScore'] = None) -> Path:
    """Write a MIDI file. Default program 65 is Alto Sax.

    Note velocity carries the confidence, so an uncertain note is quiet rather
    than absent - playing the result back is then itself a check on the
    transcription: the wrong notes are the ones you strain to hear.

    By default the notes keep their performed timing, which is what a DAW
    wants. Given `score` (the same notes quantised by `notation.quantize`),
    they are written where the score puts them instead, at its tempo and time
    signature and with bar 1 on a MIDI bar line, so a notation program imports
    the same rhythm the MusicXML shows.
    """
    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError(
            'MIDI export needs pretty_midi (pip install pretty_midi)') from exc

    if score is None:
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    else:
        tempo = float(round(score.bpm))
        # 480 ticks a beat: divisible by 12, so triplets land on exact ticks.
        midi = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=program)

    def velocity(confidence: float) -> int:
        return int(40 + 87 * max(0.0, min(1.0, confidence)))

    if score is None:
        for note in notes:
            instrument.notes.append(pretty_midi.Note(
                velocity=velocity(note.confidence),
                pitch=int(note.midi),
                start=float(note.start),
                end=float(max(note.end, note.start + 0.05)),
            ))
    else:
        midi.time_signature_changes.append(
            pretty_midi.TimeSignature(score.beats_per_bar, 4, 0.0))
        # A pickup sits at the end of MIDI bar 1, so bar lines agree.
        lead = (score.bar_ticks - score.pickup) % score.bar_ticks
        seconds_per_tick = 60.0 / tempo / score.ticks_per_beat
        for note in score.notes:
            instrument.notes.append(pretty_midi.Note(
                velocity=velocity(note.confidence),
                pitch=int(note.midi),
                start=(note.start + lead) * seconds_per_tick,
                end=(note.end + lead) * seconds_per_tick,
            ))

    midi.instruments.append(instrument)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path


def format_musicxml(notes: Sequence[TranscribedNote],
                    grid: Optional['BeatGrid'] = None, *, key=None,
                    transpose: int = 0, lyrics=None, title: str = '',
                    artist: str = '', beats_per_bar: int = 4,
                    pickup: int = 0) -> str:
    """Sheet music: the notes quantised against `grid` (see `notation`) and
    written as MusicXML (see `musicxml`). Without a trustworthy grid the
    tempo comes from the note spacing and the score says it is approximate.

    `key` is the concert key and `transpose` the semitones the notes were
    raised by, as `TranscriptionOutput` records them; `lyrics` is the
    `TimedLyrics` the notes were labelled from, which tells one held word
    from a repeated one."""
    from .musicxml import to_musicxml
    from .notation import quantize

    score = quantize(notes, grid, beats_per_bar=beats_per_bar, pickup=pickup,
                     lyrics=lyrics)
    return to_musicxml(score, key=key, transpose=transpose, title=title,
                       artist=artist)


def score_for_output(output: 'TranscriptionOutput', audio_path=None,
                     beats_per_bar: int = 4, pickup: int = 0,
                     grid: Optional['BeatGrid'] = None) -> 'QuantizedScore':
    """Quantise a finished transcription for notation.

    Against, in order: `grid` if given; the grid the rhythm stage tracked, if
    it ran; one tracked now from the source audio (`rhythm` caches it, so a
    second export is free); and, when none of those is trustworthy, the note
    spacing, with the score marked approximate. Problems finding a grid are
    in the score's `warnings`.
    """
    from .notation import grid_for_output, quantize

    warnings: List[str] = []
    if grid is None:
        grid, warnings = grid_for_output(output, audio_path)
    outcome = output.lyrics
    lyrics = outcome.lyrics if outcome is not None and outcome.found else None
    score = quantize(output.notes, grid, beats_per_bar=beats_per_bar,
                     pickup=pickup, lyrics=lyrics)
    score.warnings[:0] = warnings
    return score


def musicxml_for_output(output: 'TranscriptionOutput',
                        score: Optional['QuantizedScore'] = None,
                        title: Optional[str] = None,
                        artist: Optional[str] = None,
                        transpose: Optional[int] = None, **options) -> str:
    """The MusicXML for a finished transcription: its written key and
    transposition, its lyrics, and the track name and artist it was run
    with - each overridable. `options` go to `score_for_output` when no
    `score` is given."""
    from .musicxml import to_musicxml

    request = output.request
    if score is None:
        score = score_for_output(output, **options)
    if transpose is None:
        transpose = int(request.transpose) if request else 0
    return to_musicxml(
        score, key=output.key, transpose=transpose,
        title=title if title is not None else (
            request.track_name if request else ''),
        artist=artist if artist is not None else (
            request.artist_name if request else ''))


FORMATS = ('table', 'csv', 'json', 'leadsheet', 'lrc', 'midi', 'musicxml')


def render(notes: Sequence[TranscribedNote], fmt: str = 'table',
           **kwargs) -> str:
    """Render notes in the named format."""
    if fmt == 'table':
        return format_table(notes, show_voters=kwargs.get('show_voters', False))
    if fmt == 'csv':
        return format_csv(notes)
    if fmt == 'json':
        return format_json(notes, **{k: v for k, v in kwargs.items()
                                     if k != 'show_voters'})
    if fmt == 'leadsheet':
        return format_leadsheet(notes)
    if fmt == 'lrc':
        return format_lrc(notes, title=kwargs.get('title', ''),
                          artist=kwargs.get('artist', ''))
    if fmt == 'midi':
        raise ValueError('MIDI is binary - use write_midi() with a path')
    if fmt == 'musicxml':
        options = ('grid', 'key', 'transpose', 'lyrics', 'title', 'artist',
                   'beats_per_bar', 'pickup')
        return format_musicxml(notes, **{k: v for k, v in kwargs.items()
                                         if k in options})
    raise ValueError(f"Unknown format {fmt!r}. Available: {FORMATS}")
