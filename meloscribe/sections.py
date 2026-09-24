"""Splitting a transcription into sections a person can step through.

A whole song is several hundred notes, and nobody reads or practises a melody
that way. People think in lines of lyric and in parts of a song, so the UI
slides through those and shows - and plays - only the notes of one.

Two granularities:

- **line**: one lyric line each. When there are no lyric timings worth
  trusting, one sung phrase each instead, split at the rests.
- **part**: runs of lines separated by a long break or by a block of lyrics
  that repeats elsewhere - roughly verse / chorus / bridge. A part whose words
  repeat an earlier part says so, which is how a chorus shows itself. With
  lyrics, breaks are measured between lyric lines, and whatever is sung in
  one (stray notes, an ad-lib) is a part of its own, never the start of the
  verse after it.

The invariant the UI relies on: at each granularity every note belongs to
exactly one section, and sections are in time order, so stepping through them
visits the whole melody once. Notes that fall outside every lyric line (an
ad-lib, a hum, bleed in an instrumental break) get sections of their own
rather than being dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

# A note starting this far ahead of a line's first word still belongs to that
# line. Line starts are snapped to vocal onsets, and a pickup syllable that
# lands a hair early would otherwise end the previous line.
LINE_LEAD = 0.15

# A silence at least this long ends a phrase (no-lyrics fallback). Breaths
# between sung phrases are typically 0.2-0.6s; notes inside a phrase are
# rarely separated by more than a short rest.
PHRASE_REST = 0.45
# Phrases longer than this are split again at their longest internal rest -
# a 30-second "phrase" is not something anyone can use.
MAX_PHRASE = 10.0
# A phrase this short (in sung seconds) is almost always a fragment of its
# neighbour, or a stray note, and is merged into the nearer one.
MIN_PHRASE_SUNG = 0.6
# Gaps this close to the longest are as good a place to cut, and the one
# nearest the middle wins. Note times come in 10ms frames, so legato singing
# has runs of exactly equal gaps; taking the first of them cut one note off
# the front at a time.
GAP_TIE = 0.02

# A silence at least this long between sections starts a new part.
PART_BREAK = 2.5
# A part longer than this is split at its longest gap between lines.
MAX_PART = 45.0
# Two lines or more repeating elsewhere is structure (a chorus), not chance.
MIN_REPEAT_LINES = 2


@dataclass
class Section:
    """One contiguous stretch of the song and the notes inside it."""
    start: float
    end: float
    kind: str                       # 'lyric' | 'untexted' | 'phrase' | 'part'
    note_indices: List[int] = field(default_factory=list)
    text: str = ''
    label: str = ''
    # Timed lyric lines inside the section, for drawing them in place.
    lines: List[Tuple[float, float, str]] = field(default_factory=list)
    repeat_of: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            'label': self.label,
            'kind': self.kind,
            'start': round(self.start, 3),
            'end': round(self.end, 3),
            'text': self.text,
            'notes': list(self.note_indices),
            'lines': [{'start': round(s, 3), 'end': round(e, 3), 'text': t}
                      for s, e, t in self.lines],
            'repeat_of': self.repeat_of,
        }


@dataclass
class SongSections:
    """Both granularities, and what they were built from."""
    lines: List[Section]
    parts: List[Section]
    basis: str                      # 'lyrics' | 'phrases'
    reason: str = ''

    def to_dict(self) -> Dict:
        return {
            'basis': self.basis,
            'reason': self.reason,
            'line': [s.to_dict() for s in self.lines],
            'part': [s.to_dict() for s in self.parts],
        }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def build_sections(notes: Sequence, lyric_lines: Optional[Sequence] = None,
                   duration: float = 0.0,
                   timing_trusted: bool = True) -> SongSections:
    """Split `notes` into lyric lines (or phrases) and song parts.

    `notes` need `start`, `end` and `midi`; `lyric_lines` need `start`, `end`
    and `text`. Indices in the result refer to positions in `notes`.
    `timing_trusted=False` says the lyric line times are placeholders (plain
    lyrics spread evenly across the track) and must not be used to cut.
    """
    order = sorted(range(len(notes)), key=lambda i: notes[i].start)
    usable = [l for l in (lyric_lines or []) if _has_words(l.text)]

    if usable and timing_trusted:
        lines = _lyric_sections(notes, order, usable, duration)
        basis, reason = 'lyrics', ''
    else:
        lines = _phrase_sections(notes, order)
        basis = 'phrases'
        if usable:
            reason = ('The lyrics found have no timing of their own, so the '
                      'melody is split at its rests instead.')
        else:
            reason = 'No timed lyrics, so the melody is split at its rests.'

    _label_lines(lines)
    parts = _group_parts(notes, lines)
    return SongSections(lines=lines, parts=parts, basis=basis, reason=reason)


def sections_for_output(output) -> SongSections:
    """Sections for a finished `TranscriptionOutput`."""
    lyrics = output.lyrics
    found = lyrics is not None and lyrics.found
    lines = lyrics.lyrics.lines if found else None
    # Plain lyrics are spread evenly across the track as a placeholder; only
    # forced alignment gives them real times. Cutting the song at the
    # placeholder times would put the wrong notes under every line.
    trusted = not (found and lyrics.tier.startswith('lrclib-plain')
                   and not lyrics.lyrics.has_word_timing)
    return build_sections(output.notes, lines, duration=output.duration,
                          timing_trusted=trusted)


# --------------------------------------------------------------------------
# Lines from lyrics
# --------------------------------------------------------------------------

def _lyric_sections(notes, order: List[int], lyric_lines,
                    duration: float) -> List[Section]:
    lyric_lines = sorted(lyric_lines, key=lambda l: l.start)
    windows: List[Tuple[float, float]] = []
    for i, line in enumerate(lyric_lines):
        following = (lyric_lines[i + 1].start if i + 1 < len(lyric_lines)
                     else float('inf'))
        end = line.end if line.end is not None and line.end > line.start \
            else following
        if end == float('inf') and duration:
            end = max(duration, line.start)
        windows.append((line.start, min(end, following)))

    # Each note goes to the last line starting at or before it (allowing the
    # lead), provided that line has not already ended.
    members: List[List[int]] = [[] for _ in lyric_lines]
    loose: List[int] = []
    cursor = -1
    for idx in order:
        note = notes[idx]
        while (cursor + 1 < len(windows)
               and windows[cursor + 1][0] - LINE_LEAD <= note.start):
            cursor += 1
        line = cursor
        # The lead is for a pickup that starts early and runs on into its
        # line. A note mostly before the line starts - the short last
        # syllable of the line before - stays with that one.
        if (line > 0 and note.start < windows[line][0]
                and _overlap(note, windows[line - 1])
                > _overlap(note, windows[line])):
            line -= 1
        if line >= 0 and note.start < windows[line][1]:
            members[line].append(idx)
        else:
            loose.append(idx)

    sections: List[Section] = []
    for line, (w_start, w_end), idxs in zip(lyric_lines, windows, members):
        if idxs:
            start = min(w_start, notes[idxs[0]].start)
            # End on the last sung note, not the window: an LRC line runs
            # until the next one begins, which can be a whole guitar solo.
            end = max(notes[i].end for i in idxs)
        else:
            start = w_start
            end = w_end if w_end != float('inf') else w_start + 4.0
            end = min(end, w_start + 8.0)
        sections.append(Section(start=start, end=max(end, start), kind='lyric',
                                note_indices=idxs, text=line.text.strip(),
                                lines=[(line.start,
                                        min(w_end, max(end, line.start)),
                                        line.text.strip())]))

    # Notes outside every line keep their own sections, split at rests.
    for phrase in _phrases(notes, loose):
        sections.append(_phrase_section(notes, phrase, kind='untexted'))

    sections.sort(key=lambda s: (s.start, s.end))
    return _absorb_fragments(notes, sections, kinds=('untexted',))


def _overlap(note, window: Tuple[float, float]) -> float:
    return max(0.0, min(note.end, window[1]) - max(note.start, window[0]))


# --------------------------------------------------------------------------
# Phrases without lyrics
# --------------------------------------------------------------------------

def _phrase_sections(notes, order: List[int]) -> List[Section]:
    sections = [_phrase_section(notes, p, kind='phrase')
                for p in _phrases(notes, order)]
    return _absorb_fragments(notes, sections, kinds=('phrase',))


def _phrases(notes, idxs: List[int]) -> List[List[int]]:
    """Split time-ordered note indices at rests, then split long phrases."""
    if not idxs:
        return []
    phrases: List[List[int]] = [[idxs[0]]]
    for prev, idx in zip(idxs, idxs[1:]):
        if notes[idx].start - notes[prev].end >= PHRASE_REST:
            phrases.append([idx])
        else:
            phrases[-1].append(idx)

    out: List[List[int]] = []
    for phrase in phrases:
        out.extend(_split_long(notes, phrase))
    return out


def _split_long(notes, phrase: List[int]) -> List[List[int]]:
    # A stack, not recursion: minutes of singing without a rest is thousands
    # of notes, and this must not be what decides the recursion limit.
    out: List[List[int]] = []
    stack = [phrase]
    while stack:
        piece = stack.pop()
        cut = None
        if notes[piece[-1]].end - notes[piece[0]].start > MAX_PHRASE:
            cut = _phrase_cut(notes, piece)
        if cut is None:
            out.append(piece)
        else:
            stack.extend([piece[cut:], piece[:cut]])
    return out


def _phrase_cut(notes, piece: List[int]) -> Optional[int]:
    """Where to split a long phrase: its longest rest, among the cuts that
    leave a phrase's worth of singing on both sides.

    A sliver would be folded straight back by `_absorb_fragments`, leaving
    the phrase as long as it was.
    """
    sung = [notes[i].end - notes[i].start for i in piece]
    total, before = sum(sung), 0.0
    cuts: List[int] = []
    for k in range(1, len(piece)):
        before += sung[k - 1]
        if before >= MIN_PHRASE_SUNG and total - before >= MIN_PHRASE_SUNG:
            cuts.append(k)
    if not cuts:
        return None
    best = _widest([notes[piece[k]].start - notes[piece[k - 1]].end
                    for k in cuts],
                   [notes[piece[k]].start for k in cuts],
                   (notes[piece[0]].start + notes[piece[-1]].end) / 2)
    return cuts[best]


def _widest(gaps: List[float], times: List[float], middle: float) -> int:
    """Index of the longest gap, near-ties going to the one nearest `middle`."""
    longest = max(gaps)
    return min((k for k, gap in enumerate(gaps) if gap >= longest - GAP_TIE),
               key=lambda k: abs(times[k] - middle))


def _phrase_section(notes, idxs: List[int], kind: str) -> Section:
    return Section(start=notes[idxs[0]].start,
                   end=max(notes[i].end for i in idxs),
                   kind=kind, note_indices=list(idxs))


def _sung(notes, section: Section) -> float:
    return sum(notes[i].end - notes[i].start for i in section.note_indices)


def _absorb_fragments(notes, sections: List[Section],
                      kinds: Tuple[str, ...]) -> List[Section]:
    """Fold tiny note-only sections into the closer neighbour.

    A lone 80ms note in an instrumental break would otherwise be a stop on the
    slider of its own. Only merged when a neighbour is reasonably close; an
    isolated fragment stays, because every note must stay reachable. No merge
    may stretch a section past MAX_PHRASE either: short notes a second apart
    would otherwise chain, one merge at a time, into a single section as long
    as the whole run of them.
    """
    i = 0
    while i < len(sections):
        section = sections[i]
        target = None
        if section.kind in kinds and _sung(notes, section) < MIN_PHRASE_SUNG:
            target = _absorb_target(sections, i)
        if target is None:
            i += 1
            continue
        host = sections[target]
        host.note_indices = sorted(host.note_indices + section.note_indices,
                                   key=lambda j: notes[j].start)
        host.start = min(host.start, section.start)
        host.end = max(host.end, section.end)
        del sections[i]
        # Only the host and the section before it can have changed: look at
        # them again rather than rescanning the song.
        i = max(0, i - 1)
    return sections


def _absorb_target(sections: List[Section], i: int) -> Optional[int]:
    section = sections[i]
    neighbours = []
    if i > 0:
        neighbours.append((section.start - sections[i - 1].end, i - 1))
    if i + 1 < len(sections):
        neighbours.append((sections[i + 1].start - section.end, i + 1))
    for gap, target in sorted(neighbours):
        host = sections[target]
        if gap > PHRASE_REST * 3:
            return None
        # Folded into the line after it, a fragment makes that line start
        # early. Reach is measured from the line's own start, not from where
        # earlier merges moved it, or strays filling a break chain into the
        # verse and it seems to begin seconds before its first word.
        if (target > i and host.kind == 'lyric' and host.lines
                and host.lines[0][0] - section.start > PHRASE_REST * 3):
            continue
        span = max(host.end, section.end) - min(host.start, section.start)
        if span <= MAX_PHRASE:
            return target
    return None


def _label_lines(sections: List[Section]) -> None:
    counters = {'lyric': 0, 'phrase': 0}
    for section in sections:
        if section.kind == 'untexted':
            section.label = 'No lyric'
            continue
        counters[section.kind] = counters.get(section.kind, 0) + 1
        name = 'Line' if section.kind == 'lyric' else 'Phrase'
        section.label = f"{name} {counters[section.kind]}"


# --------------------------------------------------------------------------
# Parts
# --------------------------------------------------------------------------

def _group_parts(notes, lines: List[Section]) -> List[Section]:
    """Group line sections into song parts."""
    if not lines:
        return []

    lyric = [i for i, s in enumerate(lines) if s.kind == 'lyric']
    if lyric:
        cuts = _lyric_part_cuts(notes, lines, lyric)
    else:
        cuts = {0}
        for i in range(1, len(lines)):
            if lines[i].start - lines[i - 1].end >= PART_BREAK:
                cuts.add(i)
    cuts = _split_long_parts(lines, sorted(c for c in cuts if c < len(lines)))

    parts: List[Section] = []
    signatures: List[Tuple[str, ...]] = []
    for n, cut in enumerate(cuts):
        stop = cuts[n + 1] if n + 1 < len(cuts) else len(lines)
        members = lines[cut:stop]
        lyric = [s for s in members if s.kind == 'lyric']
        signatures.append(tuple(_normalise(s.text) for s in lyric))
        parts.append(Section(
            start=members[0].start,
            end=max(s.end for s in members),
            kind='part',
            note_indices=[i for s in members for i in s.note_indices],
            text=' / '.join(s.text for s in lyric),
            lines=[line for s in members for line in s.lines],
            label=f"Part {n + 1}"))

    for n, part in enumerate(parts):
        if signatures[n]:
            first = signatures.index(signatures[n])
            if first < n:
                part.repeat_of = first
    return parts


def _lyric_part_cuts(notes, lines: List[Section], lyric: List[int]) -> Set[int]:
    """Where parts start, decided on the lyric timeline.

    Stray notes in an instrumental break - bleed, an ad-lib - fill its
    silence, so the gaps between sections may never reach PART_BREAK even
    though the gap between the lyrics either side of the break does. Wordless
    sections never start the verse after them: singing that runs on from a
    part's last lyric without a rest ends that part, and the rest of the
    break is a part of its own, unless too little is sung in it to be worth
    a stop.
    """
    starts = {b for a, b in zip(lyric, lyric[1:])
              if lines[b].start - lines[a].end >= PART_BREAK}
    # Repeated blocks of lyric are structure: each occurrence of a repeated
    # run of lines starts a part, and the line after it starts the next one.
    # The runs are of lyric lines only, so a wordless section inside a block
    # (a held last syllable, an ad-lib) cannot break it. A line sung again
    # straight after itself counts once: 'na na na' eight times over is one
    # line held, not a block repeating itself.
    texts = [_normalise(lines[i].text) for i in lyric]
    heads = [k for k in range(len(texts)) if k == 0 or texts[k] != texts[k - 1]]
    for a, b, length in _repeated_runs([texts[k] for k in heads]):
        for g in (a, a + length, b, b + length):
            k = heads[g] if g < len(heads) else len(lyric)
            if 0 < k < len(lyric):
                starts.add(lyric[k])

    # Whatever is sung before the first word is a part of its own.
    cuts = {0, lyric[0]}
    for k, last in enumerate(lyric):
        following = lyric[k + 1] if k + 1 < len(lyric) else len(lines)
        if following < len(lines) and following not in starts:
            continue
        cuts.add(following)
        tail = last
        while (tail + 1 < following
               and lines[tail + 1].start - lines[tail].end < PHRASE_REST):
            tail += 1
        rest = range(tail + 1, following)
        if sum(_sung(notes, lines[i]) for i in rest) >= MIN_PHRASE_SUNG:
            cuts.add(tail + 1)
    return cuts


def _split_long_parts(lines: List[Section], cuts: List[int]) -> List[int]:
    """Split any part longer than MAX_PART at its longest gap between lines.

    Continuous singing with no break and no repeated block would otherwise be
    one part the length of the song, which is the same as no parts at all.
    """
    out: List[int] = []
    bounds = cuts + [len(lines)]
    stack = [(bounds[i], bounds[i + 1]) for i in range(len(cuts))][::-1]
    while stack:
        a, b = stack.pop()
        if b - a >= 2 and lines[b - 1].end - lines[a].start > MAX_PART:
            inner = range(a + 1, b)
            k = inner[_widest([lines[j].start - lines[j - 1].end for j in inner],
                              [lines[j].start for j in inner],
                              (lines[a].start + lines[b - 1].end) / 2)]
            stack.extend([(k, b), (a, k)])
        else:
            out.append(a)
    return out


def _repeated_runs(texts: List[str]) -> List[Tuple[int, int, int]]:
    """The outermost maximal runs of at least MIN_REPEAT_LINES lines that
    occur twice, as (first start, second start, length).

    A run never overlaps its own repeat. The caller collapses a line repeated
    straight after itself, since 'la la la la' would otherwise be found as a
    block ('la la') repeating itself.

    The halves of a repeated block repeat too: 'A B A B C' sung twice also
    holds 'A B' four times, and each of those would cut the chorus. So a run
    whose two occurrences both lie inside occurrences of longer runs is not
    structure of its own, and is left out.
    """
    runs = []
    n = len(texts)
    for a in range(n):
        for b in range(a + 1, n):
            if texts[a] != texts[b]:
                continue
            if a > 0 and texts[a - 1] == texts[b - 1]:
                continue  # not the start of a maximal run
            length = 0
            while (b + length < n and a + length < b
                   and texts[a + length] == texts[b + length]):
                length += 1
            if length >= MIN_REPEAT_LINES:
                runs.append((a, b, length))

    outermost = []
    longer: List[Tuple[int, int]] = []   # occurrences of runs longer than this
    for length in sorted({r[2] for r in runs}, reverse=True):
        # reach[s]: how far the longer occurrences starting at or before s go.
        reach = [0] * (n + 1)
        for start, end in longer:
            reach[start] = max(reach[start], end)
        for s in range(1, n + 1):
            reach[s] = max(reach[s], reach[s - 1])
        group = [r for r in runs if r[2] == length]
        outermost += [(a, b, length) for a, b, _ in group
                      if reach[a] < a + length or reach[b] < b + length]
        longer += [(s, s + length) for a, b, _ in group for s in (a, b)]
    return outermost


def _normalise(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def _has_words(text: str) -> bool:
    return bool(re.search(r'\w', text or ''))
