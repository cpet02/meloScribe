"""Serving audio so a browser can seek in it.

An <audio> element jumps to a section by asking for a byte range. Without
Range support it can only play from the start: every jump means downloading
the song up to that point, and some browsers refuse to seek at all.

Starlette's FileResponse answers Range requests from 0.39 on, and when it can,
it is what serves the file - a library's own implementation over a
hand-rolled one. Older Starlette, which an older FastAPI pins, returns 200
and the whole body for every request, so for those a single byte range is
served by hand.

Depends on Starlette and the standard library only, so it can be imported,
and its fallback tested, without the rest of the app.
"""

from __future__ import annotations

import mimetypes
import os
import re
from typing import Iterator, Optional, Tuple

import starlette
from starlette.responses import FileResponse, Response, StreamingResponse

CHUNK_SIZE = 64 * 1024

# One range; `bytes=a-b`, `bytes=a-` or `bytes=-n`. Several ranges at once
# are legal but no browser seeks that way, so they get the whole file.
_BYTE_RANGE = re.compile(r'^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$', re.IGNORECASE)


def serves_ranges(version: str) -> bool:
    """Whether a Starlette version's FileResponse answers Range requests."""
    match = re.match(r'(\d+)\.(\d+)', version)
    # An unreadable version gets the fallback: it works on every Starlette,
    # while trusting FileResponse on an old one silently breaks seeking.
    return bool(match) and (int(match.group(1)), int(match.group(2))) >= (0, 39)


# Read at call time, so a test can force the fallback on a new Starlette.
NATIVE_RANGES = serves_ranges(starlette.__version__)


def audio_response(path, range_header: Optional[str] = None,
                   media_type: Optional[str] = None) -> Response:
    """A response for `path` that honours a Range request on any Starlette."""
    media_type = (media_type or mimetypes.guess_type(str(path))[0]
                  or 'application/octet-stream')
    if NATIVE_RANGES:
        return FileResponse(path, media_type=media_type)

    # Advertised even on a 200, or the browser never asks for a range.
    whole = _WholeFile(path, media_type=media_type,
                       headers={'Accept-Ranges': 'bytes'})
    if not range_header:
        return whole

    size = os.stat(path).st_size
    span = parse_range(range_header, size)
    if span is None:
        return whole  # malformed: ignoring Range is always allowed
    if span == ():
        return Response(status_code=416,
                        headers={'Content-Range': f"bytes */{size}"})

    start, end = span
    return StreamingResponse(
        _read(path, start, end - start + 1), status_code=206,
        media_type=media_type,
        headers={'Content-Range': f"bytes {start}-{end}/{size}",
                 'Accept-Ranges': 'bytes',
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


class _WholeFile(FileResponse):
    """The whole file, whatever the Range header says.

    Old Starlette behaves this way anyway. A new one would act on the header
    itself, so when the fallback is forced there (in tests) it would not be
    the fallback being exercised.
    """

    async def __call__(self, scope, receive, send) -> None:
        headers = [(k, v) for k, v in scope['headers'] if k != b'range']
        await super().__call__({**scope, 'headers': headers}, receive, send)


def _read(path, start: int, length: int) -> Iterator[bytes]:
    with open(path, 'rb') as handle:
        handle.seek(start)
        while length > 0:
            chunk = handle.read(min(CHUNK_SIZE, length))
            if not chunk:
                return
            length -= len(chunk)
            yield chunk
