"""Byte-range serving for audio seeking (`meloscribe.api.media`).

The module is loaded from its file rather than imported as part of the
package, so this runs with nothing but Starlette, httpx and pytest installed.
"""

import asyncio
import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip('httpx')
from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.routing import Route  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'meloscribe_media',
    Path(__file__).resolve().parent.parent / 'meloscribe' / 'api' / 'media.py')
media = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(media)

# Several chunks long, so a range has to be stitched across chunk boundaries.
DATA = os.urandom(3 * media.CHUNK_SIZE + 1234)
SIZE = len(DATA)


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / 'song.wav'
    path.write_bytes(DATA)
    return path


def _client(path):
    """A client for a one-route app serving `path`."""
    def endpoint(req: Request):
        return media.audio_response(path, req.headers.get('range'),
                                    if_range=req.headers.get('if-range'))

    return TestClient(Starlette(routes=[Route('/audio', endpoint)]))


@pytest.fixture
def client(audio):
    return _client(audio)


def _get(client, header=None, **headers):
    if header:
        headers['Range'] = header
    return client.get('/audio', headers=headers)


@pytest.mark.parametrize('header,first,last', [
    ('bytes=10-19', 10, 19),
    ('bytes=60000-140000', 60000, 140000),       # crosses chunk boundaries
    (f'bytes={SIZE - 100}-', SIZE - 100, SIZE - 1),
    ('bytes=-50', SIZE - 50, SIZE - 1),
    (f'bytes=0-{SIZE * 2}', 0, SIZE - 1),        # end past the file: clamped
    (f'bytes={SIZE - 1}-{SIZE - 1}', SIZE - 1, SIZE - 1),
    (f'bytes=-{SIZE * 2}', 0, SIZE - 1),         # suffix longer than the file
])
def test_a_single_range_is_served_byte_exact(client, header, first, last):
    response = _get(client, header)
    assert response.status_code == 206
    assert response.content == DATA[first:last + 1]
    assert response.headers['content-range'] == f"bytes {first}-{last}/{SIZE}"
    assert response.headers['content-length'] == str(last - first + 1)
    assert response.headers['accept-ranges'] == 'bytes'
    assert response.headers['content-type'].startswith('audio/')


@pytest.mark.parametrize('header', [f'bytes={SIZE}-', f'bytes={SIZE + 5}-{SIZE + 9}',
                                    'bytes=-0'])
def test_a_range_past_the_end_is_unsatisfiable(client, header):
    response = _get(client, header)
    assert response.status_code == 416
    assert response.headers['content-range'] == f"bytes */{SIZE}"


def test_no_range_gets_the_whole_file_and_an_invitation_to_seek(client):
    response = _get(client)
    assert response.status_code == 200
    assert response.content == DATA
    assert response.headers['content-length'] == str(SIZE)
    # Without this on the 200 the browser never asks for a range at all.
    assert response.headers['accept-ranges'] == 'bytes'


@pytest.mark.parametrize('header', ['bytes=abc', 'items=0-10', 'bytes=0-1,5-6',
                                    'bytes=19-10', 'bytes=-', '0-10'])
def test_a_malformed_range_is_ignored(client, header):
    response = _get(client, header)
    assert response.status_code == 200
    assert response.content == DATA
    assert response.headers['accept-ranges'] == 'bytes'


def test_an_empty_file(tmp_path):
    empty = tmp_path / 'empty.wav'
    empty.write_bytes(b'')
    client = _client(empty)
    assert _get(client, 'bytes=0-').status_code == 416
    assert _get(client, 'bytes=-10').status_code == 416
    response = _get(client)
    assert response.status_code == 200 and response.content == b''


def test_if_range_serves_a_range_only_of_the_same_file(client):
    whole = _get(client)
    for validator in (whole.headers['etag'], whole.headers['last-modified']):
        same = _get(client, 'bytes=10-19', **{'If-Range': validator})
        assert same.status_code == 206 and same.content == DATA[10:20]
    # Changed since the client's first request: a range of it would splice
    # two different files, so the whole file comes back instead.
    stale = _get(client, 'bytes=10-19', **{'If-Range': '"not-this-file"'})
    assert stale.status_code == 200 and stale.content == DATA


def test_a_response_mid_send_does_not_hold_the_file(audio):
    """A paused browser simply stops reading. On Windows a file cannot be
    deleted or renamed while a handle is open, so serving that held one
    made re-separating the song fail; between chunks nothing may hold it."""
    response = media.audio_response(audio, 'bytes=0-')

    async def client_that_stops_reading():
        body = response.body_iterator
        first = await body.__anext__()
        # What re-separation does to the stem the browser was playing.
        os.replace(audio, audio.with_name('replaced.wav'))
        await body.aclose()
        return first

    assert asyncio.run(client_that_stops_reading()) == DATA[:media.CHUNK_SIZE]


def test_a_file_deleted_mid_response_ends_it(audio):
    chunks = media._read(audio, 0, SIZE)
    assert next(chunks) == DATA[:media.CHUNK_SIZE]
    audio.unlink()
    assert list(chunks) == []


def test_range_parsing():
    assert media.parse_range('bytes=0-0', 10) == (0, 0)
    assert media.parse_range('Bytes = 2 - 4', 10) == (2, 4)
    assert media.parse_range('bytes=7-', 10) == (7, 9)
    assert media.parse_range('bytes=-3', 10) == (7, 9)
    assert media.parse_range('bytes=10-', 10) == ()
    assert media.parse_range('bytes=0-1, 4-5', 10) is None
