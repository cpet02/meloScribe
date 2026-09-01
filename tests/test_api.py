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

from meloscribe.api.app import app  # noqa: E402

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
