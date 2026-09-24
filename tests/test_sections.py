"""Sections: lyric lines (or phrases) and song parts for the section slider.

Runs without models, network or GPU. The one property everything else rests
on is that at each granularity every note is in exactly one section, in time
order - the UI steps through the sections to visit the whole melody once, so
a dropped or doubled note is a bug the user can see.
"""

import json
import random
import time

import pytest

from meloscribe import sections as S
from meloscribe.lyrics.align import LyricLine, LyricWord, TimedLyrics
from meloscribe.lyrics.service import LyricsOutcome
from meloscribe.pipeline import TranscriptionOutput
from meloscribe.pitch.engine import TranscribedNote


def _note(start, end, midi=60):
    return TranscribedNote(midi, start, end, 0.9)


def _line(start, text, end=None):
    return LyricLine(start=start, text=text, end=end)


def _run(notes, start, count, length=0.3, step=0.4):
    """`count` notes of `length` seconds, one every `step`."""
    notes.extend(_note(start + k * step, start + k * step + length)
                 for k in range(count))
    return notes


def _members(result, grain='lines'):
    return [s.note_indices for s in getattr(result, grain)]


def _assert_partition(result, n_notes):
    for grain in (result.lines, result.parts):
        seen = sorted(i for s in grain for i in s.note_indices)
        assert seen == list(range(n_notes)), 'a note is missing or doubled'
        starts = [s.start for s in grain]
        assert starts == sorted(starts), 'sections out of time order'


# --------------------------------------------------------------------------
# The invariant
# --------------------------------------------------------------------------

def _random_layout(rng):
    notes, t = [], rng.uniform(0, 5)
    for _ in range(rng.choice([0, 1, 2, 7, 40, 150, 400])):
        t += rng.choice([0.0, 0.0, 0.01, 0.05, 0.2, 0.5, 1.0, 3.0,
                         rng.uniform(0, 8)])
        length = rng.choice([0.03, 0.1, 0.25, 0.6, 2.0, rng.uniform(0.01, 12)])
        notes.append(_note(round(t, 2), round(t + length, 2),
                           rng.randint(40, 80)))
        if rng.random() < 0.9:          # mostly monophonic, sometimes not
            t += length
    rng.shuffle(notes)

    span = max((n.end for n in notes), default=30.0) + 20
    texts = ['la la', 'oh yeah', 'chorus one', 'chorus two', 'verse', '♪',
             '', '...']
    lines = []
    for _ in range(rng.choice([0, 1, 3, 10, 40])):
        start = round(rng.uniform(-1, span), 2)
        shape = rng.random()
        end = (None if shape < 0.4
               else start - 1 if shape < 0.5            # end before start
               else start if shape < 0.55               # zero length
               else round(start + rng.uniform(0, 15), 2))  # may overlap next
        text = rng.choice(texts) + ('' if rng.random() < 0.6
                                    else str(rng.randint(0, 3)))
        lines.append(_line(start, text, end))
    if lines and rng.random() < 0.3:
        lines.append(_line(lines[0].start, 'same time', None))  # duplicate
    return notes, lines, rng.choice([0.0, span]), rng.random() < 0.8


def test_every_note_is_in_exactly_one_section_at_both_granularities():
    """Randomised: unsorted and overlapping notes, lines with missing, early,
    zero-length and overlapping ends, duplicate timestamps, wordless lines,
    lines after the last note, trusted and untrusted timing."""
    rng = random.Random(20260924)
    for _ in range(600):
        notes, lines, duration, trusted = _random_layout(rng)
        result = S.build_sections(notes, lines, duration=duration,
                                  timing_trusted=trusted)
        _assert_partition(result, len(notes))

        part_of = {i: k for k, part in enumerate(result.parts)
                   for i in part.note_indices}
        for section in result.lines + result.parts:
            # Playing a section plays its time span: its notes must be in it.
            for i in section.note_indices:
                assert section.start <= notes[i].start
                assert notes[i].end <= section.end
        for section in result.lines:
            assert len({part_of[i] for i in section.note_indices}) <= 1, \
                'a part must not split a line'
        for k, part in enumerate(result.parts):
            assert part.repeat_of is None or 0 <= part.repeat_of < k
        # The payload goes out as JSON; an infinite window end would not.
        json.dumps(result.to_dict(), allow_nan=False)


# --------------------------------------------------------------------------
# Lines from lyrics
# --------------------------------------------------------------------------

def test_notes_follow_their_lyric_lines():
    notes = [_note(10.2, 10.6), _note(11.0, 11.5), _note(14.1, 14.5)]
    result = S.build_sections(notes, [_line(10.0, 'first line'),
                                      _line(14.0, 'second line')])
    assert result.basis == 'lyrics'
    assert _members(result) == [[0, 1], [2]]
    assert [s.label for s in result.lines] == ['Line 1', 'Line 2']
    assert result.lines[0].text == 'first line'


def test_a_line_ends_on_its_last_note_not_at_the_next_line():
    """An LRC line runs until the next one starts, which can be a solo."""
    notes = [_note(10.0, 12.0), _note(40.0, 41.0)]
    result = S.build_sections(notes, [_line(10.0, 'one', end=40.0),
                                      _line(40.0, 'two')])
    assert result.lines[0].end == pytest.approx(12.0)


def test_pickup_that_runs_into_its_line_belongs_to_it():
    notes = [_note(10.0, 11.0), _note(13.9, 14.4)]
    result = S.build_sections(notes, [_line(10.0, 'one'), _line(14.0, 'two')])
    assert _members(result) == [[0], [1]]
    assert result.lines[1].start == pytest.approx(13.9)


def test_last_syllable_just_before_a_line_is_not_stolen_by_the_lead():
    """A short note wholly before the next line, inside the lead, is the end
    of the line it is sung in - here the aligned last word covers it."""
    notes = [_note(10.0, 11.0), _note(13.9, 13.98), _note(14.0, 14.5)]
    lines = [_line(10.0, 'one', end=13.99), _line(14.0, 'two', end=14.6)]
    result = S.build_sections(notes, lines)
    assert _members(result) == [[0, 1], [2]]


def test_lead_still_applies_after_the_previous_line_has_ended():
    notes = [_note(10.0, 11.0), _note(13.9, 13.98), _note(14.0, 14.5)]
    lines = [_line(10.0, 'one', end=11.5), _line(14.0, 'two')]
    result = S.build_sections(notes, lines)
    assert _members(result) == [[0], [1, 2]]


def test_notes_exactly_on_boundaries():
    notes = [_note(14.0, 14.3),        # exactly at the next line's start
             _note(21.0, 21.8),        # exactly at an explicit end: outside
             _note(10.0, 10.5)]        # exactly at a line start
    lines = [_line(10.0, 'one'), _line(14.0, 'two', end=21.0)]
    result = S.build_sections(notes, lines)
    assert _members(result) == [[2], [0], [1]]
    assert [s.kind for s in result.lines] == ['lyric', 'lyric', 'untexted']


def test_notes_outside_every_line_get_their_own_section():
    notes = _run([], 0.0, 6)                              # intro hum, 1.8 s
    notes += [_note(10.0, 11.0)]
    notes = _run(notes, 20.0, 6)                          # after the line ends
    lines = [_line(10.0, 'only line', end=11.5)]
    result = S.build_sections(notes, lines)
    assert [s.kind for s in result.lines] == ['untexted', 'lyric', 'untexted']
    assert [s.label for s in result.lines] == ['No lyric', 'Line 1', 'No lyric']
    assert _members(result) == [[0, 1, 2, 3, 4, 5], [6],
                                [7, 8, 9, 10, 11, 12]]


def test_a_stray_note_near_a_line_is_folded_into_it():
    notes = [_note(10.0, 12.0), _note(12.6, 12.7)]
    result = S.build_sections(notes, [_line(10.0, 'line', end=12.1)])
    assert _members(result) == [[0, 1]]


def test_an_isolated_stray_note_keeps_its_own_section():
    notes = [_note(10.0, 12.0), _note(30.0, 30.1)]
    result = S.build_sections(notes, [_line(10.0, 'line', end=12.1)])
    assert _members(result) == [[0], [1]]


def test_unsorted_notes_keep_their_original_indices():
    notes = [_note(14.2, 14.4), _note(10.5, 10.7), _note(10.1, 10.3)]
    result = S.build_sections(notes, [_line(10.0, 'one'), _line(14.0, 'two')])
    assert _members(result) == [[2, 1], [0]]


def test_a_line_ending_before_it_starts_runs_to_the_next_line():
    notes = [_note(10.0, 10.5), _note(12.0, 12.5)]
    result = S.build_sections(notes, [_line(10.0, 'one', end=9.0),
                                      _line(14.0, 'two')])
    assert _members(result)[0] == [0, 1]


def test_overlapping_whisper_segments_are_cut_at_the_next_start():
    notes = [_note(10.0, 10.5), _note(12.2, 12.5)]
    lines = [_line(10.0, 'one', end=13.0), _line(12.0, 'two', end=14.0)]
    result = S.build_sections(notes, lines)
    assert _members(result) == [[0], [1]]


def test_duplicate_timestamps_do_not_lose_notes():
    notes = [_note(10.0, 10.5), _note(11.0, 11.5)]
    lines = [_line(10.0, 'lead vocal'), _line(10.0, 'backing vocal')]
    result = S.build_sections(notes, lines)
    _assert_partition(result, 2)


def test_lines_after_the_last_note_are_empty_but_finite():
    notes = [_note(1.0, 1.5)]
    lines = [_line(1.0, 'sung'), _line(50.0, 'never sung')]
    for duration in (0.0, 60.0):
        result = S.build_sections(notes, lines, duration=duration)
        assert result.lines[1].note_indices == []
        assert result.lines[1].end <= 58.0
        json.dumps(result.to_dict(), allow_nan=False)


def test_wordless_lines_are_not_lines():
    notes = [_note(10.0, 10.5)]
    result = S.build_sections(notes, [_line(5.0, '♪'), _line(10.0, 'words'),
                                      _line(12.0, '  ')])
    assert [s.text for s in result.lines] == ['words']


def test_no_notes():
    result = S.build_sections([], [_line(1.0, 'a'), _line(5.0, 'b')])
    assert [s.note_indices for s in result.lines] == [[], []]
    assert S.build_sections([], None).to_dict() == {
        'basis': 'phrases', 'line': [], 'part': [],
        'reason': 'No timed lyrics, so the melody is split at its rests.'}


# --------------------------------------------------------------------------
# Phrases, without lyric timing
# --------------------------------------------------------------------------

def test_phrases_split_at_rests():
    notes = _run([], 0.0, 4)                 # ends at 1.5
    notes = _run(notes, 2.0, 4)              # a 0.5 s rest: split
    notes = _run(notes, 3.9, 4)              # a 0.4 s rest: no split
    result = S.build_sections(notes, None)
    assert result.basis == 'phrases'
    assert _members(result) == [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]]
    assert [s.label for s in result.lines] == ['Phrase 1', 'Phrase 2']


@pytest.mark.parametrize('length,step', [(0.2, 0.2), (0.25, 0.25), (0.2, 0.25)])
def test_long_legato_is_split_evenly_not_one_note_at_a_time(length, step):
    """Equal gaps are the norm in legato: notes come in 10ms frames. Taking
    the first longest gap shaved one note off per split, and folding the
    slivers back left the phrase as long as before."""
    notes = _run([], 0.0, int(30 / step), length=length, step=step)
    result = S.build_sections(notes, None)
    spans = [s.end - s.start for s in result.lines]
    assert all(span <= S.MAX_PHRASE for span in spans), spans
    assert min(spans) > 5.0, spans


def test_minutes_without_a_rest_do_not_hit_the_recursion_limit():
    notes = _run([], 0.0, 3000, length=0.25, step=0.25)
    result = S.build_sections(notes, None)
    _assert_partition(result, 3000)
    assert all(s.end - s.start <= S.MAX_PHRASE for s in result.lines)


def test_split_uses_the_longest_rest_then_the_middle():
    notes = _run([], 0.0, 12)                # 0.1 s gaps, ends at 4.7
    notes = _run(notes, 5.0, 28)             # a 0.3 s rest first; 11.1 s
    result = S.build_sections(notes, None)
    assert [len(m) for m in _members(result)] == [12, 14, 14]


def test_short_notes_a_second_apart_do_not_chain_into_one_section():
    """Each is a fragment, and each has a neighbour close enough to fold into
    - which, unchecked, glued a minute of them into one section."""
    notes = [_note(k * 1.0, k * 1.0 + 0.1) for k in range(60)]
    result = S.build_sections(notes, None)
    _assert_partition(result, 60)
    assert all(s.end - s.start <= S.MAX_PHRASE for s in result.lines)
    assert len(result.lines) >= 6


def test_a_lone_fragment_between_phrases_joins_the_nearer():
    notes = _run([], 0.0, 5)                 # 0.0 - 1.9
    notes += [_note(2.5, 2.6)]               # 0.6 s after, 1.4 s before
    notes = _run(notes, 4.0, 5)
    result = S.build_sections(notes, None)
    assert _members(result) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]


# --------------------------------------------------------------------------
# Parts
# --------------------------------------------------------------------------

def _song(texts, gap=0.5, line_length=2.0, rest_after=()):
    """One note per line; `rest_after` holds line indices followed by 3 s."""
    notes, lines, t = [], [], 0.0
    for k, text in enumerate(texts):
        lines.append(_line(t, text))
        notes.append(_note(t, t + line_length))
        t += line_length + (3.0 if k in rest_after else gap)
    return notes, lines


def test_a_long_silence_starts_a_new_part():
    notes, lines = _song(['a', 'b', 'c', 'd'], rest_after={1})
    result = S.build_sections(notes, lines)
    assert [p.note_indices for p in result.parts] == [[0, 1], [2, 3]]
    assert [p.label for p in result.parts] == ['Part 1', 'Part 2']
    assert result.parts[0].text == 'a / b'


def test_a_repeated_block_is_a_chorus():
    texts = ['verse one', 'verse two', 'Chorus, a!', 'chorus b',
             'verse three', 'verse four', 'chorus A', 'Chorus b',
             'bridge', 'chorus a', 'chorus b']
    notes, lines = _song(texts)
    result = S.build_sections(notes, lines)
    assert [p.note_indices for p in result.parts] == [
        [0, 1], [2, 3], [4, 5], [6, 7], [8], [9, 10]]
    assert [p.repeat_of for p in result.parts] == [None, None, None, 1, None, 1]


def test_one_repeated_line_is_not_structure():
    notes, lines = _song(['hook', 'a', 'b', 'hook', 'c'])
    result = S.build_sections(notes, lines)
    assert len(result.parts) == 1


def test_a_line_repeated_in_a_row_is_not_its_own_repeat():
    notes, lines = _song(['la', 'la', 'la'])
    result = S.build_sections(notes, lines)
    assert len(result.parts) == 1 and result.parts[0].repeat_of is None


def test_a_long_part_is_split_at_its_longest_gap():
    notes, lines = _song(['l%d' % k for k in range(24)])       # 60 s, even
    lines[12].start += 0.4                                     # a breath here
    notes[12].start += 0.4
    notes[12].end += 0.4
    result = S.build_sections(notes, lines)
    assert [p.note_indices[0] for p in result.parts] == [0, 12]


def test_a_long_part_with_even_gaps_is_split_near_the_middle():
    notes, lines = _song(['l%d' % k for k in range(40)])       # 100 s
    result = S.build_sections(notes, lines)
    sizes = [len(p.note_indices) for p in result.parts]
    assert all(p.end - p.start <= S.MAX_PART for p in result.parts)
    assert min(sizes) >= 8, sizes


def _verses_around_a_break(strays):
    """Verse one closed by a rest marker at 8 s, `strays` (start, end) in the
    instrumental break, verse two from 18 s."""
    notes, lines = [], []
    for k, t in enumerate((0.0, 3.0, 6.0)):
        lines.append(_line(t, 'verse one %d' % k, end=8.0 if k == 2 else None))
        notes.append(_note(t, t + 2.0))
    notes += [_note(start, end, 70) for start, end in strays]
    for k, t in enumerate((18.0, 21.0, 24.0)):
        lines.append(_line(t, 'verse two %d' % k))
        notes.append(_note(t, t + 2.0))
    return notes, lines


def test_stray_notes_in_a_break_do_not_join_the_next_verse():
    """Regression: bleed and ad-libs filled the break, so no gap between
    sections reached PART_BREAK, and the break was glued onto the next verse
    - its part began 7 s before its first word."""
    strays = [(t, t + 0.3) for t in (11.0, 12.2, 13.4, 14.6, 15.8, 17.0)]
    notes, lines = _verses_around_a_break(strays)
    result = S.build_sections(notes, lines, duration=30.0)
    _assert_partition(result, len(notes))

    first, gap, second = result.parts
    assert first.text.startswith('verse one') and first.end == pytest.approx(8.0)
    assert gap.text == '' and gap.note_indices == [3, 4, 5, 6, 7, 8]
    assert second.text.startswith('verse two')
    assert second.start == pytest.approx(18.0), 'starts at its first word'


def test_a_little_stray_singing_in_a_break_stays_with_the_part_before():
    notes, lines = _verses_around_a_break([(13.0, 13.2)])
    result = S.build_sections(notes, lines, duration=30.0)
    assert [p.note_indices for p in result.parts] == [[0, 1, 2, 3], [4, 5, 6]]
    assert result.parts[1].start == pytest.approx(18.0)


def test_singing_that_runs_on_from_the_last_line_ends_its_part():
    """A melisma past the aligned end of the last word, then the break."""
    notes, lines = _verses_around_a_break([(8.1, 9.0), (9.0, 9.8)])
    result = S.build_sections(notes, lines, duration=30.0)
    assert [p.note_indices for p in result.parts] == [[0, 1, 2, 3, 4],
                                                      [5, 6, 7]]


def test_singing_before_the_first_lyric_is_a_part_of_its_own():
    notes, lines = _verses_around_a_break([])
    notes += [_note(-6.0, -5.0), _note(-4.9, -4.0)]    # an intro, 4 s ahead
    result = S.build_sections(notes, lines, duration=30.0)
    assert result.parts[0].note_indices == [6, 7]
    assert result.parts[1].start == pytest.approx(0.0)


def test_strays_before_a_verse_cannot_pull_its_start_seconds_early():
    """Folding fragments forward, one at a time, once moved a line's start
    6 s ahead of its first word."""
    strays = [(5.02, 5.14), (6.3, 6.38), (7.27, 7.35), (8.17, 8.41),
              (9.69, 9.74), (10.22, 10.48), (11.82, 11.92), (12.7, 12.76),
              (13.36, 13.47), (14.68, 14.77), (15.41, 15.47), (16.81, 16.97)]
    notes, lines = _verses_around_a_break(strays)
    result = S.build_sections(notes, lines, duration=30.0)
    verse_two = next(s for s in result.lines if s.text == 'verse two 0')
    assert verse_two.start >= 18.0 - S.PHRASE_REST * 3
    assert result.parts[-1].start >= 18.0 - S.PHRASE_REST * 3


def test_phrase_parts_still_break_at_silences():
    notes = _run([], 0.0, 5)                          # ends at 1.9
    notes = _run(notes, 4.5, 5)                       # 2.6 s later: new part
    notes = _run(notes, 8.8, 5)                       # 2.4 s later: same part
    result = S.build_sections(notes, None)
    assert [len(p.note_indices) for p in result.parts] == [5, 10]


def test_phrases_have_no_repeats():
    notes, _ = _song(['a'] * 6)
    result = S.build_sections(notes, None)
    assert all(p.repeat_of is None for p in result.parts)


def test_a_150_line_song_builds_quickly():
    texts = []
    for block in range(30):
        texts += (['chorus a', 'chorus b', 'chorus c'] if block % 2
                  else ['verse %d %d' % (block, k) for k in range(7)])
    notes, lines = _song(texts[:150])
    started = time.perf_counter()
    result = S.build_sections(notes, lines)
    assert time.perf_counter() - started < 1.0
    _assert_partition(result, 150)
    started = time.perf_counter()
    S._repeated_runs(['la'] * 150)
    assert time.perf_counter() - started < 1.0


# --------------------------------------------------------------------------
# From a pipeline output
# --------------------------------------------------------------------------

def _output(tier, words=False, lines=None):
    lines = lines or [LyricLine(start=1.0, text='one two', end=2.0),
                      LyricLine(start=5.0, text='three four', end=6.0)]
    lyrics = TimedLyrics(lines=lines,
                         words=[LyricWord('one', 1.0, 1.5)] if words else [])
    notes = [_note(1.0, 1.5), _note(5.0, 5.5)]
    return TranscriptionOutput(notes=notes, duration=10.0,
                               lyrics=LyricsOutcome(lyrics=lyrics, tier=tier))


@pytest.mark.parametrize('tier,words,basis', [
    ('lrclib-synced', False, 'lyrics'),
    ('lrclib-synced+snapped', False, 'lyrics'),
    ('lrclib-synced+aligned', True, 'lyrics'),
    ('lrclib-plain', False, 'phrases'),
    ('lrclib-plain+snapped', False, 'phrases'),
    ('lrclib-plain+aligned', True, 'lyrics'),
    ('whisper', True, 'lyrics'),
])
def test_only_real_lyric_timings_are_used_to_cut(tier, words, basis):
    """Plain lyrics are spread evenly as a placeholder; snapping those to
    onsets does not make them real. Only alignment does."""
    result = S.sections_for_output(_output(tier, words))
    assert result.basis == basis
    if basis == 'phrases':
        assert 'no timing' in result.reason


def test_outputs_without_lyrics_fall_back_to_phrases():
    output = _output('lrclib-synced')
    output.lyrics = None
    assert S.sections_for_output(output).basis == 'phrases'

    empty = _output('not-found')
    empty.lyrics.lyrics = None
    assert S.sections_for_output(empty).basis == 'phrases'


def test_payload_shape():
    payload = S.sections_for_output(_output('lrclib-synced')).to_dict()
    assert set(payload) == {'basis', 'reason', 'line', 'part'}
    first = payload['line'][0]
    assert set(first) == {'label', 'kind', 'start', 'end', 'text', 'notes',
                          'lines', 'repeat_of'}
    assert first['lines'] == [{'start': 1.0, 'end': 1.5, 'text': 'one two'}]
    # 3.5 s of silence between the lines: two parts.
    assert [p['notes'] for p in payload['part']] == [[0], [1]]
