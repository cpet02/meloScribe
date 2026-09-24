"""Getting lyrics onto notes, at the finest timing resolution available.

Three tiers, best first:

1. **Synced LRC from LRClib, refined.** Line timestamps are hand-made and
   routinely 200-500ms adrift. Snapping each line to the nearest detected
   vocal onset removes most of that error at almost no cost.

2. **Forced alignment.** When only plain lyrics exist, a CTC acoustic model
   aligns the known text to the vocal stem and produces *word-level* timings.
   This is better than any LRC file: line-level timing can only say which
   phrase a note belongs to, while word timings put a syllable on a note.

3. **Transcribe from scratch.** With no lyrics at all, Whisper produces text
   and timings together. Least accurate, but it is the difference between
   some lyrics and none.

Both alignment tiers are optional dependencies. Missing ones degrade to the
next tier down rather than failing the run.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# [mm:ss.xx] or [mm:ss.xxx], optionally several on one line.
LRC_TIMESTAMP = re.compile(r'\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]')
LRC_METADATA = re.compile(r'^\[(ti|ar|al|by|offset|length|re|ve):', re.IGNORECASE)


@dataclass
class LyricLine:
    """One timed line of lyrics."""
    start: float
    text: str
    end: Optional[float] = None
    words: List['LyricWord'] = field(default_factory=list)

    @property
    def word_texts(self) -> List[str]:
        return [w for w in re.split(r'\s+', self.text.strip()) if w]


@dataclass
class LyricWord:
    """One word with its own timing, from forced alignment."""
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class TimedLyrics:
    """A whole lyric sheet, however it was produced."""
    lines: List[LyricLine]
    words: List[LyricWord] = field(default_factory=list)
    source: str = 'unknown'
    language: str = ''

    @property
    def has_word_timing(self) -> bool:
        return bool(self.words)

    def to_lrc(self, title: str = '', artist: str = '') -> str:
        """Render back to LRC, so generated timings can be saved and reused."""
        out: List[str] = []
        if title:
            out.append(f"[ti:{title}]")
        if artist:
            out.append(f"[ar:{artist}]")
        for line in sorted(self.lines, key=lambda x: x.start):
            minutes, seconds = divmod(max(0.0, line.start), 60)
            out.append(f"[{int(minutes):02d}:{seconds:05.2f}]{line.text}")
        return '\n'.join(out)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def parse_lrc(content: str) -> List[LyricLine]:
    """Parse LRC text into timed lines.

    Handles repeated timestamps on one line (`[00:12.00][01:30.00]chorus`),
    which is how LRC files mark a repeated refrain and which a naive parser
    silently drops.

    A timestamp with no words (`[01:23.45]`, or just `♪`) is not a line but it
    is not noise either: it marks where the line before it stops, which is how
    LRC files mark an instrumental break. It closes that line instead of being
    dropped, or the line would stretch across the whole break.
    """
    lines: List[LyricLine] = []
    breaks: List[float] = []

    for raw in content.splitlines():
        raw = raw.strip()
        if not raw or LRC_METADATA.match(raw):
            continue

        stamps = list(LRC_TIMESTAMP.finditer(raw))
        if not stamps:
            continue

        text = raw[stamps[-1].end():].strip()
        times = [_stamp_seconds(stamp) for stamp in stamps]
        if not re.search(r'\w', text):
            breaks.extend(times)  # a timing marker with no words: a rest
            continue

        lines.extend(LyricLine(start=t, text=text) for t in times)

    lines.sort(key=lambda x: x.start)
    breaks.sort()
    for i, line in enumerate(lines):
        following = lines[i + 1].start if i + 1 < len(lines) else None
        idx = bisect.bisect_right(breaks, line.start)
        rest = breaks[idx] if idx < len(breaks) else None
        if rest is not None and (following is None or rest < following):
            line.end = rest
        else:
            line.end = following
    return lines


def _stamp_seconds(stamp: 're.Match') -> float:
    minutes = int(stamp.group(1))
    seconds = int(stamp.group(2))
    fraction = stamp.group(3) or '0'
    # Two digits are hundredths, three are milliseconds.
    divisor = 100.0 if len(fraction) <= 2 else 1000.0
    return minutes * 60 + seconds + int(fraction) / divisor


def parse_plain(content: str, duration: float) -> List[LyricLine]:
    """Turn untimed lyrics into evenly spaced lines.

    A deliberately poor placeholder: it exists so the pipeline has something to
    align, and should always be replaced by forced alignment when available.
    """
    texts = [line.strip() for line in content.splitlines() if line.strip()]
    if not texts:
        return []
    spacing = duration / (len(texts) + 1)
    return [LyricLine(start=(i + 1) * spacing, text=text)
            for i, text in enumerate(texts)]


# --------------------------------------------------------------------------
# Tier 1: refining existing line timings against the audio
# --------------------------------------------------------------------------

def vocal_onsets(audio_path, backtrack: bool = True) -> np.ndarray:
    """Onset times in the vocal stem - where phrases actually begin."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    return librosa.onset.onset_detect(y=y, sr=sr, units='time',
                                      backtrack=backtrack)


def refine_line_times(lines: Sequence[LyricLine], onsets: np.ndarray,
                      max_shift: float = 0.6) -> List[LyricLine]:
    """Snap each line's start to the nearest vocal onset.

    `max_shift` is a guard rail, not a preference: a line whose nearest onset
    is further away than this is left where the LRC put it, on the grounds
    that we have probably found an unrelated onset rather than the true start
    of the phrase. Moving a line to the wrong place is worse than leaving it
    slightly early.
    """
    if len(onsets) == 0:
        return list(lines)

    onsets = np.asarray(onsets, dtype=float)
    ordered = sorted(lines, key=lambda x: x.start)
    refined: List[Tuple[LyricLine, bool]] = []

    for i, line in enumerate(ordered):
        # An end short of the next line's start was set on purpose - by a rest
        # marker in the LRC - and must survive the re-timing below.
        following = ordered[i + 1].start if i + 1 < len(ordered) else None
        explicit = (line.end is not None and line.end > line.start
                    and (following is None or line.end < following))
        nearest = onsets[int(np.argmin(np.abs(onsets - line.start)))]
        start = float(nearest) if abs(nearest - line.start) <= max_shift else line.start
        if explicit and start >= line.end:
            # An onset past the point where the line stops is not its start.
            # Taking it would lose the end: recomputed, it would run the line
            # through the break, or for the last line fall before the start.
            start = line.start
        refined.append((LyricLine(start=start, text=line.text, end=line.end,
                                  words=list(line.words)), explicit))

    refined.sort(key=lambda pair: pair[0].start)
    for i, (line, explicit) in enumerate(refined):
        if i + 1 < len(refined):
            following = refined[i + 1][0].start
            line.end = min(line.end, following) if explicit else following
    return [line for line, _ in refined]


# --------------------------------------------------------------------------
# Tier 2: forced alignment
# --------------------------------------------------------------------------

class ForcedAligner:
    """Word-level alignment of known text to audio, via torchaudio's CTC
    forced-alignment API and the multilingual MMS model.

    Forced alignment is a much easier problem than transcription: the words are
    already known, so the model only has to decide *when* each was sung. That
    is why this beats Whisper's timings even though Whisper is the larger
    model.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self._bundle = None

    def available(self) -> bool:
        try:
            from torchaudio.pipelines import MMS_FA  # noqa: F401
            return True
        except (ImportError, AttributeError):
            return False

    def align(self, audio_path, text: str) -> List[LyricWord]:
        """Align a block of lyrics to a vocal stem, returning word timings."""
        import torch
        import torchaudio
        from torchaudio.pipelines import MMS_FA as bundle

        device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')

        waveform, sr = torchaudio.load(str(audio_path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr,
                                                      bundle.sample_rate)

        words = self._normalise(text)
        if not words:
            return []

        model = bundle.get_model().to(device)
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()

        with torch.inference_mode():
            emission, _ = model(waveform.to(device))
            token_spans = aligner(emission[0], tokenizer(words))

        # Emission frames are coarser than samples; this ratio converts back.
        ratio = waveform.shape[1] / emission.shape[1] / bundle.sample_rate

        aligned: List[LyricWord] = []
        for word, spans in zip(words, token_spans):
            if not spans:
                continue
            aligned.append(LyricWord(
                text=word,
                start=float(spans[0].start * ratio),
                end=float(spans[-1].end * ratio),
                confidence=float(np.mean([s.score for s in spans])),
            ))
        return aligned

    @staticmethod
    def _normalise(text: str) -> List[str]:
        """Lower-case, strip punctuation, drop anything unpronounceable.

        The MMS tokenizer only knows letters and apostrophes; leaving in
        punctuation or bracketed stage directions like "[Chorus]" makes the
        alignment fail on tokens that were never sung.
        """
        cleaned = re.sub(r'\[[^\]]*\]', ' ', text)
        cleaned = re.sub(r"[^\w\s']", ' ', cleaned.lower())
        return [w for w in cleaned.split() if w and not w.isdigit()]


# --------------------------------------------------------------------------
# Tier 3: transcribe from nothing
# --------------------------------------------------------------------------

class WhisperTranscriber:
    """faster-whisper fallback for tracks with no lyrics on LRClib."""

    def __init__(self, model_size: str = 'base', device: Optional[str] = None):
        self.model_size = model_size
        self.device = device

    def available(self) -> bool:
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            return True
        except ImportError:
            return False

    def transcribe(self, audio_path) -> TimedLyrics:
        from faster_whisper import WhisperModel

        device = self.device
        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'

        model = WhisperModel(self.model_size, device=device,
                             compute_type='float16' if device == 'cuda' else 'int8')
        segments, info = model.transcribe(str(audio_path), word_timestamps=True)

        lines: List[LyricLine] = []
        words: List[LyricWord] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            lines.append(LyricLine(start=float(segment.start), text=text,
                                   end=float(segment.end)))
            for word in (segment.words or []):
                words.append(LyricWord(text=word.word.strip(),
                                       start=float(word.start),
                                       end=float(word.end),
                                       confidence=float(word.probability)))

        return TimedLyrics(lines=lines, words=words, source='whisper',
                           language=getattr(info, 'language', ''))


# --------------------------------------------------------------------------
# Attaching lyrics to notes
# --------------------------------------------------------------------------

def attach_to_notes(notes: Sequence, lyrics: Optional[TimedLyrics]) -> None:
    """Write `lyric` (and `syllable` where known) onto each note, in place.

    With word timings, each note takes the word it overlaps most - which
    handles melisma correctly, since several notes can legitimately share one
    word. With only line timings, a note takes whichever line's window
    contains it, and no syllable is claimed. Notes with no lyric keep None
    rather than inheriting a neighbour's word.
    """
    if lyrics is None:
        for note in notes:
            note.lyric = None
        return

    if lyrics.has_word_timing:
        _attach_words(notes, lyrics.words)
    else:
        _attach_lines(notes, lyrics.lines)


def _attach_words(notes: Sequence, words: Sequence[LyricWord]) -> None:
    if not words:
        for note in notes:
            note.lyric = None
        return

    starts = np.array([w.start for w in words])
    ends = np.array([w.end for w in words])

    for note in notes:
        overlap = np.minimum(ends, note.end) - np.maximum(starts, note.start)
        best = int(np.argmax(overlap))
        if overlap[best] > 0:
            note.lyric = words[best].text
            note.syllable = words[best].text
        else:
            # No overlap: a sung note between words (a hum, or a held vowel
            # past the word boundary). Left unlabelled rather than guessed.
            note.lyric = None
            note.syllable = None


def _attach_lines(notes: Sequence, lines: Sequence[LyricLine]) -> None:
    if not lines:
        for note in notes:
            note.lyric = None
        return

    starts = np.array([line.start for line in lines])
    for note in notes:
        idx = int(np.searchsorted(starts, note.start, side='right') - 1)
        if idx < 0:
            note.lyric = None
            continue
        line = lines[idx]
        within = line.end is None or note.start < line.end
        note.lyric = line.text if within else None
        note.syllable = None
