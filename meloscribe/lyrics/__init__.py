"""Lyric lookup and time-syncing."""

from .align import (ForcedAligner, LyricLine, LyricWord, TimedLyrics,
                    WhisperTranscriber, attach_to_notes, parse_lrc)
from .lrclib import LrcLibClient, LyricsResult, TrackQuery, describe_track
from .service import (LyricsMode, LyricsOutcome, LyricsService,
                      MissingTrackName, resolve_lyrics)

__all__ = [
    'ForcedAligner', 'LyricLine', 'LyricWord', 'TimedLyrics',
    'WhisperTranscriber', 'attach_to_notes', 'parse_lrc',
    'LrcLibClient', 'LyricsResult', 'TrackQuery', 'describe_track',
    'LyricsMode', 'LyricsOutcome', 'LyricsService', 'MissingTrackName',
    'resolve_lyrics',
]
