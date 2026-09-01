"""End-to-end lyric resolution: from a track name to timed words.

This is where the tiers in `align.py` are actually chosen between, and where
the "a track name is required" rule lives. That rule exists because lyric
lookup without a name silently degrades into guesswork - a filename like
`track03.mp3` yields a search for "track03", which either finds nothing or,
worse, finds the wrong song and aligns its words confidently onto the melody.
Failing loudly up front is the honest behaviour; `LyricsMode.OFF` is the
supported way to skip it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from .align import (ForcedAligner, LyricLine, TimedLyrics, WhisperTranscriber,
                    parse_lrc, parse_plain, refine_line_times, vocal_onsets)
from .lrclib import LrcLibClient, LrcLibError, LyricsResult, TrackQuery

ProgressFn = Callable[[float, str], None]

# Below this mean word confidence, an alignment is treated as failed rather
# than trusted. Uncalibrated - chosen to catch gross mismatches (an
# instrumental, or the wrong song's lyrics), which measured around 0.05,
# rather than to discriminate between good and merely adequate alignments.
MIN_ALIGNMENT_CONFIDENCE = 0.15


class LyricsMode(str, Enum):
    """How hard to try for lyrics."""
    OFF = 'off'            # skip entirely - the bypass switch
    LOOKUP = 'lookup'      # LRClib only, use its timings as given
    ALIGN = 'align'        # LRClib + forced alignment to word level (default)
    TRANSCRIBE = 'transcribe'  # ...and fall back to Whisper if nothing is found


class MissingTrackName(ValueError):
    """Raised when lyrics are requested without a usable track name."""


@dataclass
class LyricsOutcome:
    """What the lyric stage managed to produce, and how."""
    lyrics: Optional[TimedLyrics] = None
    record: Optional[LyricsResult] = None
    mode: LyricsMode = LyricsMode.ALIGN
    tier: str = 'none'
    warnings: List[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.lyrics is not None and bool(self.lyrics.lines)

    def summary(self) -> str:
        if not self.found:
            return f"no lyrics ({self.tier})"
        detail = 'word-level' if self.lyrics.has_word_timing else 'line-level'
        return f"{len(self.lyrics.lines)} lines, {detail} ({self.tier})"


class LyricsService:
    """Finds lyrics for a track and times them against its vocal stem."""

    def __init__(self, client: Optional[LrcLibClient] = None,
                 aligner: Optional[ForcedAligner] = None,
                 transcriber: Optional[WhisperTranscriber] = None):
        self.client = client or LrcLibClient()
        self.aligner = aligner or ForcedAligner()
        self.transcriber = transcriber or WhisperTranscriber()

    @staticmethod
    def require_track_name(query: Optional[TrackQuery],
                           mode: LyricsMode) -> None:
        """Enforce the naming gate before any expensive work begins.

        Called before stemming, not after: discovering that a five-minute
        separation was pointless is a bad way to learn the track name was
        missing.
        """
        if mode == LyricsMode.OFF:
            return
        if query is None or not query.is_usable():
            raise MissingTrackName(
                'A track name is required to look up lyrics. Provide one, or '
                'disable lyrics (mode "off") to transcribe without them.')

    def resolve(self, vocals_path, query: TrackQuery,
                mode: LyricsMode = LyricsMode.ALIGN,
                duration: Optional[float] = None,
                progress: Optional[ProgressFn] = None) -> LyricsOutcome:
        """Fetch and time the lyrics for one track."""
        outcome = LyricsOutcome(mode=mode)
        if mode == LyricsMode.OFF:
            outcome.tier = 'disabled'
            return outcome

        self.require_track_name(query, mode)
        if duration and not query.duration:
            query.duration = duration

        if progress:
            progress(0.1, 'searching LRClib')

        try:
            record = self.client.find(query)
        except LrcLibError as exc:
            # A network failure must not sink a transcription that is otherwise
            # fine - the notes are the primary product, lyrics are a garnish.
            outcome.warnings.append(f"LRClib unavailable: {exc}")
            record = None

        outcome.record = record

        if record is not None and record.instrumental:
            outcome.tier = 'instrumental'
            outcome.warnings.append('LRClib lists this track as instrumental.')
            return outcome

        if record is None:
            return self._transcribe_fallback(vocals_path, outcome, mode, progress)

        if record.has_synced:
            outcome.lyrics = TimedLyrics(lines=parse_lrc(record.synced_lyrics),
                                         source='lrclib-synced')
            outcome.tier = 'lrclib-synced'
        elif record.has_plain:
            outcome.lyrics = TimedLyrics(
                lines=parse_plain(record.plain_lyrics, duration or 0.0),
                source='lrclib-plain')
            outcome.tier = 'lrclib-plain'
        else:
            return self._transcribe_fallback(vocals_path, outcome, mode, progress)

        if mode == LyricsMode.LOOKUP:
            return outcome

        return self._improve_timing(vocals_path, outcome, progress)

    def _improve_timing(self, vocals_path, outcome: LyricsOutcome,
                        progress: Optional[ProgressFn]) -> LyricsOutcome:
        """Upgrade whatever timings we have, as far as the tools allow."""
        text = ' '.join(line.text for line in outcome.lyrics.lines)

        if self.aligner.available() and text.strip():
            if progress:
                progress(0.5, 'forced alignment')
            try:
                words = self.aligner.align(vocals_path, text)
                confidence = (float(np.mean([w.confidence for w in words]))
                              if words else 0.0)

                if words and confidence < MIN_ALIGNMENT_CONFIDENCE:
                    # Forced alignment always returns *something*: it places
                    # the words it was given whether or not they were sung.
                    # Measured on an instrumental with invented lyrics, it
                    # produced plausible-looking timings at confidence ~0.05.
                    # Accepting that would put confident, entirely fictional
                    # word timings on the notes, so a weak alignment is
                    # rejected in favour of the coarser but honest fallback.
                    outcome.warnings.append(
                        f"Forced alignment confidence too low "
                        f"({confidence:.2f}) - the lyrics may not match this "
                        f"audio. Falling back to line timings.")
                    words = []
                elif not words:
                    outcome.warnings.append(
                        'Forced alignment produced no words; keeping LRC '
                        'timings.')

                if words:
                    outcome.lyrics.words = words
                    outcome.lyrics.lines = _lines_from_words(
                        words, outcome.lyrics.lines)
                    outcome.tier += '+aligned'
                    return outcome
            except Exception as exc:
                outcome.warnings.append(f"Forced alignment failed: {exc}")
        else:
            outcome.warnings.append(
                'Forced aligner unavailable - falling back to onset-snapped '
                'line timings (install torchaudio for word-level timing).')

        # Fallback: snap the existing line timings to vocal onsets. Cheaper and
        # coarser than alignment, but still removes the bulk of the offset that
        # hand-made LRC files carry.
        if progress:
            progress(0.6, 'snapping lines to vocal onsets')
        try:
            outcome.lyrics.lines = refine_line_times(outcome.lyrics.lines,
                                                     vocal_onsets(vocals_path))
            outcome.tier += '+snapped'
        except Exception as exc:
            outcome.warnings.append(f"Onset refinement failed: {exc}")

        return outcome

    def _transcribe_fallback(self, vocals_path, outcome: LyricsOutcome,
                             mode: LyricsMode,
                             progress: Optional[ProgressFn]) -> LyricsOutcome:
        if mode != LyricsMode.TRANSCRIBE:
            outcome.tier = 'not-found'
            outcome.warnings.append(
                'No lyrics on LRClib. Use mode "transcribe" to generate them '
                'from the audio.')
            return outcome

        if not self.transcriber.available():
            outcome.tier = 'not-found'
            outcome.warnings.append(
                'No lyrics found and faster-whisper is not installed.')
            return outcome

        if progress:
            progress(0.4, 'transcribing vocals with Whisper')
        try:
            outcome.lyrics = self.transcriber.transcribe(vocals_path)
            outcome.tier = 'whisper'
        except Exception as exc:
            outcome.tier = 'not-found'
            outcome.warnings.append(f"Whisper transcription failed: {exc}")

        return outcome


def _lines_from_words(words, original_lines: List[LyricLine]) -> List[LyricLine]:
    """Re-time the original lines from the aligned words.

    The line *text* is kept exactly as the lyric sheet had it - punctuation,
    capitalisation and all - while the timings come from alignment. Rebuilding
    lines out of the normalised alignment tokens instead would hand back
    lower-cased, punctuation-stripped lyrics, which is a poor thing to display.
    """
    if not words or not original_lines:
        return original_lines

    lines: List[LyricLine] = []
    cursor = 0
    for line in original_lines:
        count = len(line.word_texts)
        if count == 0 or cursor >= len(words):
            continue
        span = words[cursor:cursor + count]
        if not span:
            break
        lines.append(LyricLine(start=span[0].start, text=line.text,
                               end=span[-1].end, words=list(span)))
        cursor += count

    return lines or original_lines


def resolve_lyrics(vocals_path, track_name: str, artist_name: str = '',
                   mode: LyricsMode = LyricsMode.ALIGN,
                   duration: Optional[float] = None) -> LyricsOutcome:
    """Convenience entry point."""
    return LyricsService().resolve(
        vocals_path,
        TrackQuery(track_name=track_name, artist_name=artist_name,
                   duration=duration),
        mode=mode, duration=duration)
