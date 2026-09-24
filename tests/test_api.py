"""API integration tests.

These run the real FastAPI app against the real job store, including one
end-to-end transcription. The transcription case uses a short synthetic clip
with `vocals_only` set, so it exercises the whole request/job/download path in
seconds without invoking source separation.
"""

import time

import numpy as np
import pytest
import soundfile as sf

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient  # noqa: E402

from meloscribe.api import app as app_module, media  # noqa: E402
from meloscribe.api.app import app  # noqa: E402
from meloscribe.api.jobs import JobStatus  # noqa: E402
from meloscribe.lyrics.align import LyricLine, TimedLyrics  # noqa: E402
from meloscribe.lyrics.service import LyricsOutcome  # noqa: E402
from meloscribe.pipeline import TranscriptionOutput  # noqa: E402
from meloscribe.pitch.engine import TranscribedNote  # noqa: E402
from meloscribe.stems import StemResult  # noqa: E402

client = TestClient(app)


@pytest.fixture(scope='module')
def clip(tmp_path_factory):
    """A short two-note WAV, enough to produce a real transcription."""
    sr = 22050
    path = tmp_path_factory.mktemp('audio') / 'clip.wav'

    audio = np.concatenate([
        _tone(midi, 0.6, sr) for midi in (60, 64)
    ])
    sf.write(str(path), audio, sr)
    return path


def _tone(midi, duration, sr):
    t = np.arange(int(duration * sr)) / sr
    hz = 440.0 * 2 ** ((midi - 69) / 12)
    wave = sum(0.7 ** h * np.sin(2 * np.pi * hz * (h + 1) * t) for h in range(6))
    envelope = np.minimum(1.0, np.minimum(t, duration - t) * 40)
    return 0.7 * wave / np.max(np.abs(wave)) * envelope


def _upload(clip):
    with open(clip, 'rb') as handle:
        response = client.post('/api/upload',
                               files={'file': ('clip.wav', handle, 'audio/wav')})
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------
# Health and upload
# --------------------------------------------------------------------------

def test_health_reports_capabilities():
    body = client.get('/api/health').json()
    assert 'device' in body and 'voters' in body
    # The UI relies on these keys to warn about missing voters up front.
    assert isinstance(body['voters'], dict) and body['voters']


def test_index_serves_the_ui():
    response = client.get('/')
    assert response.status_code == 200
    assert 'meloScribe' in response.text


def test_upload_returns_duration_and_suggestions(clip):
    body = _upload(clip)
    assert body['upload_id'] and body['duration'] == pytest.approx(1.2, abs=0.1)
    assert 'track_name' in body['suggested']


def test_upload_rejects_unsupported_types(tmp_path):
    bad = tmp_path / 'notes.txt'
    bad.write_text('not audio')
    with open(bad, 'rb') as handle:
        response = client.post('/api/upload',
                               files={'file': ('notes.txt', handle, 'text/plain')})
    assert response.status_code == 400
    assert 'Unsupported' in response.json()['detail']


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_job_without_track_name_is_rejected(clip):
    """The naming gate must fire before any work is queued."""
    upload = _upload(clip)
    response = client.post('/api/jobs', json={'upload_id': upload['upload_id'],
                                              'track_name': ''})
    assert response.status_code == 422
    assert 'track name is required' in response.json()['detail'].lower()


def test_no_lyrics_bypasses_the_name_requirement(clip):
    upload = _upload(clip)
    response = client.post('/api/jobs', json={
        'upload_id': upload['upload_id'], 'track_name': '',
        'lyrics_mode': 'off', 'vocals_only': True})
    assert response.status_code == 200, response.text


def test_unknown_upload_id_is_404():
    response = client.post('/api/jobs', json={'upload_id': 'nope.mp3',
                                              'lyrics_mode': 'off'})
    assert response.status_code == 404


def test_path_traversal_upload_id_is_rejected():
    """A crafted id must not escape the upload directory."""
    response = client.post('/api/jobs',
                           json={'upload_id': '../../../etc/passwd',
                                 'lyrics_mode': 'off'})
    assert response.status_code in (400, 404)


@pytest.mark.parametrize('upload_id', ['../uploads_private/x.wav', '', '.'])
def test_upload_id_cannot_name_a_sibling_directory_or_the_folder_itself(upload_id):
    # A string-prefix check accepted all of these: a sibling folder sharing
    # the name's prefix, and the upload folder itself.
    response = client.post('/api/jobs',
                           json={'upload_id': upload_id, 'lyrics_mode': 'off'})
    assert response.status_code == 400


def test_unknown_lyrics_mode_is_rejected(clip):
    upload = _upload(clip)
    response = client.post('/api/jobs', json={
        'upload_id': upload['upload_id'], 'track_name': 'x',
        'lyrics_mode': 'telepathy'})
    assert response.status_code == 400


def test_missing_job_is_404():
    assert client.get('/api/jobs/deadbeef').status_code == 404


def test_downloading_an_unfinished_job_conflicts(clip):
    """Results must not be served before they exist."""
    upload = _upload(clip)
    job = client.post('/api/jobs', json={
        'upload_id': upload['upload_id'], 'lyrics_mode': 'off',
        'vocals_only': True}).json()

    # Racy by nature: either the job has not finished (409) or it already has.
    response = client.get(f"/api/jobs/{job['id']}/notes")
    assert response.status_code in (200, 409)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def _run_to_completion(clip, timeout=180):
    upload = _upload(clip)
    job = client.post('/api/jobs', json={
        'upload_id': upload['upload_id'], 'track_name': 'Test Clip',
        'lyrics_mode': 'off', 'vocals_only': True,
        'voters': ['harmonic_template']}).json()

    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state['status'] in ('done', 'failed', 'cancelled'):
            return state
        time.sleep(0.4)
    pytest.fail('job did not finish within the timeout')


@pytest.mark.slow
def test_end_to_end_job_produces_notes(clip):
    state = _run_to_completion(clip)
    assert state['status'] == 'done', state.get('error')
    assert state['progress'] == 1.0

    body = client.get(f"/api/jobs/{state['id']}/notes").json()
    assert body['notes'], 'transcription produced no notes'

    first = body['notes'][0]
    assert {'note', 'midi', 'start_time', 'confidence'} <= set(first)
    assert 0.0 <= first['confidence'] <= 1.0


@pytest.mark.slow
@pytest.mark.parametrize('fmt,expected', [
    ('csv', 'index,note'),
    ('json', '"notes"'),
    ('leadsheet', ''),
    ('lrc', ''),
])
def test_downloads_render_every_text_format(clip, fmt, expected):
    state = _run_to_completion(clip)
    response = client.get(f"/api/jobs/{state['id']}/download/{fmt}")
    assert response.status_code == 200
    assert expected in response.text


@pytest.mark.slow
def test_midi_download_is_a_real_midi_file(clip):
    state = _run_to_completion(clip)
    response = client.get(f"/api/jobs/{state['id']}/download/midi")
    assert response.status_code == 200
    assert response.content[:4] == b'MThd', 'not a MIDI header'


@pytest.mark.slow
def test_unknown_download_format_is_rejected(clip):
    state = _run_to_completion(clip)
    assert client.get(
        f"/api/jobs/{state['id']}/download/sibelius").status_code == 400


# --------------------------------------------------------------------------
# Sections and audio, on a finished job put straight into the store
# --------------------------------------------------------------------------
# Nothing is transcribed here, so these need neither the pitch voters nor
# separation - only the API's own handling of a finished result.

SONG = bytes(range(256)) * 400


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    """An upload directory of the test's own."""
    directory = tmp_path / 'uploads'
    directory.mkdir()
    monkeypatch.setattr(app_module, 'UPLOAD_DIR', directory)
    return directory


def _inject_job(uploads, vocals=None, vocals_only=False, transpose=0,
                status=JobStatus.DONE):
    (uploads / 'song.wav').write_bytes(SONG)
    notes = [TranscribedNote(60, 10.2, 10.6, 0.9),
             TranscribedNote(62, 11.0, 11.5, 0.9),
             TranscribedNote(64, 14.1, 14.5, 0.9),
             TranscribedNote(65, 30.0, 31.0, 0.9)]     # sung after every line
    lines = [LyricLine(start=10.0, text='first line', end=14.0),
             LyricLine(start=14.0, text='second line', end=15.0)]
    result = TranscriptionOutput(
        notes=notes, duration=40.0,
        lyrics=LyricsOutcome(lyrics=TimedLyrics(lines=lines),
                             tier='lrclib-synced+snapped'))
    if vocals is not None:
        result.stems = StemResult(stems={'vocals': vocals}, model='test',
                                  device='cpu', cached=True)
    job = app_module.jobs.create(filename='song.wav', params={
        'upload_id': 'song.wav', 'transpose': transpose,
        'vocals_only': vocals_only})
    job.result = result
    job.status = status
    return job


def test_notes_payload_carries_sections_and_audio(uploads, tmp_path):
    vocals = tmp_path / 'vocals.wav'
    vocals.write_bytes(b'vocal stem')
    job = _inject_job(uploads, vocals=vocals, transpose=-2)

    body = client.get(f"/api/jobs/{job.id}/notes").json()
    sections = body['sections']
    assert sections['basis'] == 'lyrics'
    for grain in ('line', 'part'):
        seen = sorted(i for s in sections[grain] for i in s['notes'])
        assert seen == list(range(len(body['notes']))), grain
    assert [s['label'] for s in sections['line']] == ['Line 1', 'Line 2',
                                                      'No lyric']
    assert body['audio'] == ['mix', 'vocals']
    assert body['transpose'] == -2
    assert body['word_level'] is False


def test_audio_without_a_range_is_the_whole_file(uploads):
    job = _inject_job(uploads)
    response = client.get(f"/api/jobs/{job.id}/audio/mix")
    assert response.status_code == 200
    assert response.content == SONG
    assert response.headers['accept-ranges'] == 'bytes'
    assert response.headers['content-type'].startswith('audio/')


@pytest.mark.parametrize('native', [True, False])
def test_audio_range_request_gets_exactly_those_bytes(uploads, monkeypatch,
                                                      native):
    """How the browser seeks to a section, on either Starlette code path."""
    if native and not media.NATIVE_RANGES:
        pytest.skip('this Starlette cannot serve ranges itself')
    monkeypatch.setattr(media, 'NATIVE_RANGES', native)
    job = _inject_job(uploads)
    response = client.get(f"/api/jobs/{job.id}/audio/mix",
                          headers={'Range': 'bytes=1000-1999'})
    assert response.status_code == 206
    assert response.content == SONG[1000:2000]
    assert response.headers['content-range'] == f"bytes 1000-1999/{len(SONG)}"


def test_vocal_stem_is_served_when_separation_made_one(uploads, tmp_path):
    vocals = tmp_path / 'vocals.wav'
    vocals.write_bytes(b'vocal stem')
    job = _inject_job(uploads, vocals=vocals)
    response = client.get(f"/api/jobs/{job.id}/audio/vocals")
    assert response.status_code == 200 and response.content == b'vocal stem'


def test_an_isolated_vocal_upload_is_its_own_vocals(uploads):
    job = _inject_job(uploads, vocals_only=True)
    response = client.get(f"/api/jobs/{job.id}/audio/vocals")
    assert response.status_code == 200 and response.content == SONG


def test_missing_vocals_are_404(uploads, tmp_path):
    unseparated = _inject_job(uploads)
    assert client.get(
        f"/api/jobs/{unseparated.id}/audio/vocals").status_code == 404

    evicted = _inject_job(uploads, vocals=tmp_path / 'cleared-from-cache.wav')
    assert client.get(f"/api/jobs/{evicted.id}/audio/vocals").status_code == 404
    assert client.get(f"/api/jobs/{evicted.id}/notes").json()['audio'] == ['mix']


def test_unknown_audio_source_is_404(uploads):
    job = _inject_job(uploads)
    for source in ('drums', 'song.wav', '..%2F..%2Fetc%2Fpasswd'):
        assert client.get(
            f"/api/jobs/{job.id}/audio/{source}").status_code == 404, source


def test_audio_never_comes_from_outside_the_upload_directory(uploads):
    job = _inject_job(uploads)
    job.params['upload_id'] = '../../../../../../etc/passwd'
    assert client.get(f"/api/jobs/{job.id}/audio/mix").status_code == 404
    assert client.get(f"/api/jobs/{job.id}/notes").json()['audio'] == []


def test_audio_of_an_unfinished_job_conflicts(uploads):
    job = _inject_job(uploads, status=JobStatus.RUNNING)
    assert client.get(f"/api/jobs/{job.id}/audio/mix").status_code == 409
    assert client.get('/api/jobs/deadbeef/audio/mix').status_code == 404
