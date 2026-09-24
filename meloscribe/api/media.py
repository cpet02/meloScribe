"""Serving audio so a browser can seek in it.

An <audio> element jumps to a section by asking for a byte range. Without
Range support it can only play from the start: every jump means downloading
the song up to that point, and some browsers refuse to seek at all.

Ranges are served here rather than by Starlette's FileResponse (which answers
them from 0.39 on), for a Windows reason: FileResponse keeps the file open
until the whole response is sent, and a browser that pauses playback simply
stops reading, so the handle can stay open for as long as the tab does. While
any handle is open Windows refuses to delete or rename the file, and
re-separating a song whose vocal stem had been playing failed - the stem cache
could not replace its entry. Here the file is opened for each chunk and closed
again before the chunk is sent, so nothing holds it while the client is not
reading.

Depends on Starlette and the standard library only, so it can be imported and
tested without the rest of the app.
"""

from __future__ import annotations

import mimetypes
import os
import re
from email.utils import formatdate
from typing import Dict, Iterator, Optional, Tuple

from starlette.responses import Response, StreamingResponse

CHUNK_SIZE = 64 * 1024

# The formats uploads come in. mimetypes alone differs by machine: on Windows
# it reads the registry, which calls '.aac' 'audio/vnd.dlna.adts'.
AUDIO_TYPES = {'.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.flac': 'audio/flac',
               '.m4a': 'audio/mp4', '.ogg': 'audio/ogg', '.opus': 'audio/ogg',
               '.aac': 'audio/aac'}

# One range; `bytes=a-b`, `bytes=a-` or `bytes=-n`. Several ranges at once
# are legal but no browser seeks that way, so they get the whole file.
_BYTE_RANGE = re.compile(r'^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$', re.IGNORECASE)


def audio_response(path, range_header: Optional[str] = None,
                   media_type: Optional[str] = None,
                   if_range: Optional[str] = None) -> Response:
    """A response for `path` that honours a single-range Range request.

    `if_range` is the request's If-Range header: a range is served only if
    it names the file as it is now, or a client resuming a download could
    stitch bytes of two different files together.
    """
    media_type = (media_type or AUDIO_TYPES.get(os.path.splitext(str(path))[1].lower())
                  or mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
    stat = os.stat(path)
    size = stat.st_size
    validators = _validators(stat)
    # Advertised even on a 200, or the browser never asks for a range.
    headers = {'Accept-Ranges': 'bytes', **validators}

    span = parse_range(range_header, size) if range_header else None
    if span is not None and if_range is not None \
            and if_range.strip() not in validators.values():
        span = None  # changed since the client's first request: all of it

    if span is None:
        # No range, a malformed one (ignoring Range is always allowed), or a
        # stale If-Range.
        return StreamingResponse(_read(path, 0, size, stat), media_type=media_type,
                                 headers={**headers, 'Content-Length': str(size)})
    if span == ():
        return Response(status_code=416,
                        headers={'Content-Range': f"bytes */{size}"})

    start, end = span
    return StreamingResponse(
        _read(path, start, end - start + 1, stat), status_code=206,
        media_type=media_type,
        headers={**headers, 'Content-Range': f"bytes {start}-{end}/{size}",
                 'Content-Length': str(end - start + 1)})


def parse_range(header: str, size: int) -> Optional[Tuple[int, ...]]:
    """The inclusive (first, last) byte a Range header asks for.

    None when the header is malformed, or asks for more than one range; an
    empty tuple when it is well-formed but asks for nothing the file has.
    """
    match = _BYTE_RANGE.match(header)
    if match is None:
        return None
    first, last = match.groups()
    if not first and not last:
        return None
    if not first:
        # `-n`: the final n bytes, or the whole file when it is shorter.
        length = int(last)
        if length == 0 or size == 0:
            return ()
        return (max(0, size - length), size - 1)
    first_byte = int(first)
    last_byte = int(last) if last else size - 1
    if last and last_byte < first_byte:
        return None
    if first_byte >= size:
        return ()
    return (first_byte, min(last_byte, size - 1))


def _validators(stat: os.stat_result) -> Dict[str, str]:
    """ETag and Last-Modified for the file as it is now: what If-Range and a
    browser's cache compare against."""
    return {'ETag': f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"',
            'Last-Modified': formatdate(stat.st_mtime, usegmt=True)}


def _read(path, start: int, length: int,
          served: Optional[os.stat_result] = None) -> Iterator[bytes]:
    """Bytes [start, start + length) of `path`, a chunk at a time.

    The file is open only while a chunk is read, never while the chunk waits
    for the client to take it (see the module docstring). Reopened by name,
    it may meanwhile have been deleted or replaced - re-separation renames
    new stems into place - so each chunk is checked against `served`, the
    file the response's headers describe, and the response ends short
    rather than splice the new file's bytes under the old one's ETag and
    length. The client asks again, and If-Range sorts out which file it gets.
    """
    position = start
    while length > 0:
        try:
            with open(path, 'rb') as handle:
                if served is not None:
                    now = os.fstat(handle.fileno())
                    if (now.st_size, now.st_mtime_ns) != (served.st_size,
                                                          served.st_mtime_ns):
                        return
                handle.seek(position)
                chunk = handle.read(min(CHUNK_SIZE, length))
        except OSError:
            # Deleted (FileNotFoundError), or being deleted on Windows
            # (PermissionError): end it here as well.
            return
        if not chunk:
            return
        position += len(chunk)
        length -= len(chunk)
        yield chunk
