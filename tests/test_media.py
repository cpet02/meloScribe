"""Byte-range serving for audio seeking (`meloscribe.api.media`).

The module is loaded from its file rather than imported as part of the
package, so this runs with nothing but Starlette, httpx and pytest installed.
That is how the fallback is checked against an old Starlette, whose
FileResponse ignores Range.
"""

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


@pytest.fixture(params=['fallback', 'native'])
def client(request, audio, monkeypatch):
    """A client for a one-route app serving `audio`, on each code path."""
    if request.param == 'native':
        if not media.NATIVE_RANGES:
            pytest.skip('this Starlette cannot serve ranges itself')
    else:
        monkeypatch.setattr(media, 'NATIVE_RANGES', False)

    def endpoint(req: Request):
        return media.audio_response(audio, req.headers.get('range'))

    return TestClient(Starlette(routes=[Route('/audio', endpoint)]))


@pytest.fixture
def fallback(audio, monkeypatch):
    monkeypatch.setattr(media, 'NATIVE_RANGES', False)

    def endpoint(req: Request):
        return media.audio_response(audio, req.headers.get('range'))

    return TestClient(Starlette(routes=[Route('/audio', endpoint)]))


def _get(client, header=None):
    return client.get('/audio', headers={'Range': header} if header else {})


# --------------------------------------------------------------------------
# Behaviour both paths share
# --------------------------------------------------------------------------
# Only what seeking needs. Starlette's own handling, 0.39 to at least 0.41.3,
# differs at the edges no browser seeks with - a 416 whose Content-Range
# lacks its unit, and a 416 for a suffix longer than the file - so those are
# checked on the fallback alone.

@pytest.mark.parametrize('header,first,last', [
    ('bytes=10-19', 10, 19),
    ('bytes=60000-140000', 60000, 140000),       # crosses chunk boundaries
    (f'bytes={SIZE - 100}-', SIZE - 100, SIZE - 1),
    ('bytes=-50', SIZE - 50, SIZE - 1),
    (f'bytes=0-{SIZE * 2}', 0, SIZE - 1),        # end past the file: clamped
    (f'bytes={SIZE - 1}-{SIZE - 1}', SIZE - 1, SIZE - 1),
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
    assert _get(client, header).status_code == 416


def test_no_range_gets_the_whole_file_and_an_invitation_to_seek(client):
    response = _get(client)
    assert response.status_code == 200
    assert response.content == DATA
    # Without this on the 200 the browser never asks for a range at all.
    assert response.headers['accept-ranges'] == 'bytes'


# --------------------------------------------------------------------------
# The fallback's own rules
# --------------------------------------------------------------------------

@pytest.mark.parametrize('header', [f'bytes={SIZE}-', 'bytes=-0'])
def test_fallback_says_how_long_the_file_is_when_refusing(fallback, header):
    response = _get(fallback, header)
    assert response.status_code == 416
    assert response.headers['content-range'] == f"bytes */{SIZE}"


def test_fallback_serves_all_of_a_suffix_longer_than_the_file(fallback):
    response = _get(fallback, f'bytes=-{SIZE * 2}')
    assert response.status_code == 206
    assert response.content == DATA
    assert response.headers['content-range'] == f"bytes 0-{SIZE - 1}/{SIZE}"

@pytest.mark.parametrize('header', ['bytes=abc', 'items=0-10', 'bytes=0-1,5-6',
                                    'bytes=19-10', 'bytes=-', '0-10'])
def test_fallback_ignores_a_malformed_range(fallback, header):
    response = _get(fallback, header)
    assert response.status_code == 200
    assert response.content == DATA
    assert response.headers['accept-ranges'] == 'bytes'


def test_fallback_on_an_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(media, 'NATIVE_RANGES', False)
    empty = tmp_path / 'empty.wav'
    empty.write_bytes(b'')
    app = Starlette(routes=[Route('/audio', lambda req: media.audio_response(
        empty, req.headers.get('range')))])
    client = TestClient(app)
    assert _get(client, 'bytes=0-').status_code == 416
    assert _get(client, 'bytes=-10').status_code == 416
    assert _get(client).status_code == 200


def test_range_parsing():
    assert media.parse_range('bytes=0-0', 10) == (0, 0)
    assert media.parse_range('Bytes = 2 - 4', 10) == (2, 4)
    assert media.parse_range('bytes=7-', 10) == (7, 9)
    assert media.parse_range('bytes=-3', 10) == (7, 9)
    assert media.parse_range('bytes=10-', 10) == ()
    assert media.parse_range('bytes=0-1, 4-5', 10) is None


def test_the_library_is_used_when_it_can_do_ranges():
    assert not media.serves_ranges('0.37.2')
    assert not media.serves_ranges('0.38.6')
    assert media.serves_ranges('0.39.0')
    assert media.serves_ranges('0.41.3')
    assert media.serves_ranges('1.7.0')
    assert not media.serves_ranges('unknown')   # the fallback always works


def test_seeking_works_on_the_installed_starlette(audio):
    """Whatever Starlette is installed, the default path answers a range."""
    app = Starlette(routes=[Route('/audio', lambda req: media.audio_response(
        audio, req.headers.get('range')))])
    response = _get(TestClient(app), 'bytes=100-199')
    assert response.status_code == 206
    assert response.content == DATA[100:200]
