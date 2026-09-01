"""LRClib client.

LRClib publishes a documented JSON API, so this is a client, not a scraper -
which is both kinder to their servers and far less fragile than parsing HTML
that can be restyled at any time.

Two endpoints matter:

    /api/get     exact lookup by track, artist, album and duration
    /api/search  fuzzy text search when the exact lookup misses

Duration is the field that does the real work. Song titles collide constantly -
live versions, remasters, covers, radio edits - and the runtime is what
distinguishes them. Every result is therefore checked against the actual
duration of the audio file, and a mismatched hit is rejected rather than
returned as a near-enough answer that would misalign the whole transcription.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

API_BASE = 'https://lrclib.net/api'

# LRClib asks clients to identify themselves and link back to the project.
USER_AGENT = 'meloScribe/1.0 (https://github.com/cpet02/meloScribe)'

# A hit whose duration differs from the audio by more than this is a different
# recording, however well the title matches.
DURATION_TOLERANCE_S = 3.0


class LrcLibError(RuntimeError):
    """Raised when the API cannot be reached or returns something unusable."""


@dataclass
class LyricsResult:
    """One LRClib record."""
    id: Optional[int]
    track_name: str
    artist_name: str
    album_name: str = ''
    duration: float = 0.0
    instrumental: bool = False
    plain_lyrics: str = ''
    synced_lyrics: str = ''
    source: str = 'lrclib'

    @property
    def has_synced(self) -> bool:
        return bool(self.synced_lyrics and self.synced_lyrics.strip())

    @property
    def has_plain(self) -> bool:
        return bool(self.plain_lyrics and self.plain_lyrics.strip())

    def describe(self) -> str:
        kind = ('synced' if self.has_synced
                else 'plain' if self.has_plain
                else 'instrumental' if self.instrumental else 'empty')
        return (f"{self.artist_name} - {self.track_name} "
                f"[{self.duration:.0f}s, {kind}]")

    @classmethod
    def from_api(cls, payload: Dict) -> 'LyricsResult':
        return cls(
            id=payload.get('id'),
            track_name=payload.get('trackName') or '',
            artist_name=payload.get('artistName') or '',
            album_name=payload.get('albumName') or '',
            duration=float(payload.get('duration') or 0.0),
            instrumental=bool(payload.get('instrumental')),
            plain_lyrics=payload.get('plainLyrics') or '',
            synced_lyrics=payload.get('syncedLyrics') or '',
        )


@dataclass
class TrackQuery:
    """What we know about the track we are looking for."""
    track_name: str
    artist_name: str = ''
    album_name: str = ''
    duration: Optional[float] = None

    def is_usable(self) -> bool:
        return bool(self.track_name and self.track_name.strip())


class LrcLibClient:
    """Minimal, polite LRClib client built on the standard library."""

    def __init__(self, base_url: str = API_BASE, timeout: float = 10.0,
                 retries: int = 2, user_agent: str = USER_AGENT):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retries = retries
        self.user_agent = user_agent

    def _get(self, endpoint: str, params: Dict[str, str]):
        url = f"{self.base_url}/{endpoint}?" + urllib.parse.urlencode(
            {k: v for k, v in params.items() if v not in (None, '')})
        request = urllib.request.Request(
            url, headers={'User-Agent': self.user_agent,
                          'Accept': 'application/json'})

        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.HTTPError as exc:
                # 404 is a normal "no such track", not a failure worth retrying.
                if exc.code == 404:
                    return None
                last_error = exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < self.retries:
                time.sleep(0.5 * (attempt + 1))  # brief backoff

        raise LrcLibError(f"LRClib request failed ({url}): {last_error}")

    def get_exact(self, query: TrackQuery) -> Optional[LyricsResult]:
        """Exact lookup. Returns None when LRClib has no matching record."""
        payload = self._get('get', {
            'track_name': query.track_name,
            'artist_name': query.artist_name,
            'album_name': query.album_name,
            'duration': str(int(query.duration)) if query.duration else '',
        })
        return LyricsResult.from_api(payload) if payload else None

    def search(self, query: TrackQuery, limit: int = 10) -> List[LyricsResult]:
        """Fuzzy search, best matches first."""
        params = ({'track_name': query.track_name, 'artist_name': query.artist_name}
                  if query.artist_name else
                  {'q': query.track_name})

        payload = self._get('search', params)
        if not payload:
            return []
        return [LyricsResult.from_api(item) for item in payload[:limit]]

    def find(self, query: TrackQuery,
             duration_tolerance: float = DURATION_TOLERANCE_S,
             require_synced: bool = False) -> Optional[LyricsResult]:
        """Best available match for a track, exact lookup first.

        Ranking prefers synced lyrics over plain, and a close duration match
        over a distant one. Anything outside the duration tolerance is dropped
        outright: lyrics from a different cut of the song are worse than no
        lyrics at all, because they will silently misalign every line.
        """
        if not query.is_usable():
            raise ValueError('A track name is required to search for lyrics')

        exact = self.get_exact(query)
        if exact and self._duration_ok(exact, query, duration_tolerance):
            if not require_synced or exact.has_synced:
                return exact

        candidates = [
            result for result in self.search(query)
            if self._duration_ok(result, query, duration_tolerance)
            and (result.has_synced if require_synced else
                 (result.has_synced or result.has_plain))
        ]
        if not candidates:
            return None

        def rank(result: LyricsResult):
            gap = (abs(result.duration - query.duration)
                   if query.duration else 0.0)
            return (not result.has_synced, gap)

        return sorted(candidates, key=rank)[0]

    @staticmethod
    def _duration_ok(result: LyricsResult, query: TrackQuery,
                     tolerance: float) -> bool:
        if not query.duration or not result.duration:
            return True  # nothing to check against; let ranking decide
        return abs(result.duration - query.duration) <= tolerance


# --------------------------------------------------------------------------
# Track metadata
# --------------------------------------------------------------------------

# "01 - Artist - Title", "Artist - Title", "Artist_-_Title" and friends.
_LEADING_TRACK_NUMBER = re.compile(r'^\s*\d{1,3}\s*[-._)]\s*')
_SEPARATOR = re.compile(r'\s+-\s+|\s+_-_\s+')


def metadata_from_tags(audio_path) -> Optional[TrackQuery]:
    """Read artist/title from the file's own tags, when mutagen is installed.

    Tags are far more reliable than a filename, so they are tried first - but
    plenty of files have none, hence the filename fallback below.
    """
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return None

    try:
        tags = MutagenFile(str(audio_path), easy=True)
    except Exception:
        return None
    if tags is None:
        return None

    def first(key: str) -> str:
        value = tags.get(key)
        return str(value[0]) if value else ''

    def clean_artist(name: str) -> str:
        """Strip YouTube's auto-generated ' - Topic' suffix.

        Files pulled from YouTube's auto-generated artist channels carry it in
        the artist tag, and it makes the LRClib artist match fail outright.
        """
        return re.sub(r'\s*-\s*Topic\s*$', '', name, flags=re.IGNORECASE).strip()

    title = first('title')
    if not title:
        return None

    return TrackQuery(
        track_name=title,
        artist_name=clean_artist(first('artist')),
        album_name=first('album'),
        duration=float(getattr(tags.info, 'length', 0.0) or 0.0),
    )


def metadata_from_filename(audio_path) -> TrackQuery:
    """Best-effort artist/title from the filename."""
    stem = Path(audio_path).stem
    stem = _LEADING_TRACK_NUMBER.sub('', stem)
    stem = stem.replace('_', ' ').strip()

    parts = _SEPARATOR.split(stem, maxsplit=1)
    if len(parts) == 2:
        return TrackQuery(track_name=parts[1].strip(),
                          artist_name=parts[0].strip())
    return TrackQuery(track_name=stem)


def describe_track(audio_path, duration: Optional[float] = None,
                   original_name: Optional[str] = None) -> TrackQuery:
    """Everything we can infer about a file without asking the user.

    The result is a *suggestion*: the UI prefills its fields with this and the
    user confirms or corrects before the run starts. Guessing silently is how
    you end up with a beautifully aligned set of the wrong song's lyrics.

    `original_name` matters for uploads. The server stores them under a
    generated id, so falling back to the *stored* filename yields a track name
    like 'a1900292bf3a' - which is not obviously wrong to a caller, passes the
    "a name was provided" gate, and sends a hex string to LRClib.
    """
    query = metadata_from_tags(audio_path)
    if query is None:
        query = metadata_from_filename(original_name or audio_path)
    if duration and not query.duration:
        query.duration = duration
    return query
