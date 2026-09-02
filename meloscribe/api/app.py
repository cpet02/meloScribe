"""FastAPI backend.

Upload a file, confirm the track name, start a job, watch it progress, download
the result. The API is deliberately thin: it validates input, hands work to the
job store, and serialises what the pipeline returns.

    uvicorn meloscribe.api.app:app --reload
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..lyrics.lrclib import (LrcLibClient, LrcLibError, TrackQuery,
                             describe_track)
from ..lyrics.service import LyricsMode
from ..output import format_csv, format_json, format_leadsheet, format_lrc, \
    format_table, write_midi
from ..pipeline import Pipeline, TranscriptionRequest
from ..stems import best_device
from .jobs import JobStatus, JobStore

UPLOAD_DIR = Path('data/uploads')
EXPORT_DIR = Path('data/exports')
WEB_DIR = Path(__file__).resolve().parent.parent / 'web'

# Anything larger is almost certainly not a single song, and unbounded uploads
# are how a local tool fills a disk by accident.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_SUFFIXES = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.opus', '.aac'}

app = FastAPI(title='meloScribe', version=__version__)

# The UI is served from this same origin, but a permissive policy keeps a
# separate front-end dev server workable.
app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_methods=['*'],
    allow_headers=['*'],
)

jobs = JobStore(max_workers=1)
pipeline = Pipeline()
lrclib = LrcLibClient()


class JobRequest(BaseModel):
    """Parameters for starting a transcription."""
    upload_id: str
    track_name: str = ''
    artist_name: str = ''
    lyrics_mode: str = LyricsMode.ALIGN.value
    preset: str = 'balanced'
    transpose: int = 0
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    vocals_only: bool = False
    voters: Optional[List[str]] = None
    assess_rhythm: bool = False
    force: bool = False


def _upload_path(upload_id: str) -> Path:
    """Resolve an upload id to a path, refusing anything that escapes the
    upload directory."""
    # Traversal guard: an id like '../../etc/passwd' must not resolve outside
    # UPLOAD_DIR, even though ids are server-generated today.
    candidate = (UPLOAD_DIR / upload_id).resolve()
    if not str(candidate).startswith(str(UPLOAD_DIR.resolve())):
        raise HTTPException(status_code=400, detail='Invalid upload id')
    if not candidate.exists():
        raise HTTPException(status_code=404, detail='Upload not found')
    return candidate


@app.get('/', response_class=HTMLResponse)
def index() -> str:
    page = WEB_DIR / 'index.html'
    if not page.exists():
        return '<h1>meloScribe</h1><p>UI not found. API is at /docs.</p>'
    return page.read_text(encoding='utf-8')


@app.get('/api/health')
def health() -> Dict[str, Any]:
    """Report what is actually available, so the UI can warn up front rather
    than failing halfway through a five-minute job."""
    from ..lyrics.align import ForcedAligner, WhisperTranscriber
    from ..pitch.voters import _REGISTRY

    available = {}
    for name, cls in _REGISTRY.items():
        try:
            available[name] = cls().available()
        except Exception:
            available[name] = False

    from ..pitch.voters import resolve_voter_names

    return {
        'version': __version__,
        'device': best_device(),
        # 'voters' is what could run; 'active' is what actually will. They
        # differ because the best set depends on whether CREPE is installed,
        # and reporting only availability implies every voter is in use.
        'voters': available,
        'active_voters': list(resolve_voter_names()),
        'forced_aligner': ForcedAligner().available(),
        'whisper': WhisperTranscriber().available(),
    }


@app.post('/api/upload')
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Store an uploaded audio file and return what we can guess about it."""
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. "
                   f"Supported: {sorted(ALLOWED_SUFFIXES)}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Keep the original name for display, but store under a unique id so two
    # uploads of 'track01.mp3' cannot overwrite each other.
    import uuid
    upload_id = f"{uuid.uuid4().hex[:12]}{suffix}"
    destination = UPLOAD_DIR / upload_id

    size = 0
    with open(destination, 'wb') as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB")
            out.write(chunk)

    duration = None
    try:
        import soundfile as sf
        duration = float(sf.info(str(destination)).duration)
    except Exception:
        try:
            import librosa
            duration = float(librosa.get_duration(path=str(destination)))
        except Exception:
            pass

    # Tags come from the stored file, but the *name* fallback must use what
    # the user uploaded - the stored file is named by a generated id.
    guess = describe_track(destination, duration=duration,
                           original_name=file.filename)

    return {
        'upload_id': upload_id,
        'filename': file.filename,
        'size_bytes': size,
        'duration': duration,
        # A suggestion for the form, not a decision - the user confirms it.
        'suggested': {'track_name': guess.track_name,
                      'artist_name': guess.artist_name},
    }


@app.get('/api/lyrics/search')
def search_lyrics(track: str, artist: str = '',
                  duration: Optional[float] = None) -> Dict[str, Any]:
    """Search LRClib so the user can confirm the right version before running."""
    if not track.strip():
        raise HTTPException(status_code=400, detail='A track name is required')

    query = TrackQuery(track_name=track, artist_name=artist, duration=duration)
    try:
        results = lrclib.search(query)
    except LrcLibError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {'results': [
        {'id': r.id, 'track_name': r.track_name, 'artist_name': r.artist_name,
         'album_name': r.album_name, 'duration': r.duration,
         'instrumental': r.instrumental, 'synced': r.has_synced,
         'plain': r.has_plain,
         'duration_delta': (abs(r.duration - duration)
                            if duration and r.duration else None)}
        for r in results]}


@app.post('/api/jobs')
def create_job(request: JobRequest) -> Dict[str, Any]:
    """Validate and queue a transcription."""
    source = _upload_path(request.upload_id)

    try:
        mode = LyricsMode(request.lyrics_mode)
    except ValueError:
        raise HTTPException(status_code=400,
                            detail=f"Unknown lyrics mode {request.lyrics_mode!r}")

    # The naming gate, enforced before the job is even queued so the user is
    # told immediately rather than watching a job fail minutes later.
    if mode != LyricsMode.OFF and not request.track_name.strip():
        raise HTTPException(
            status_code=422,
            detail='A track name is required unless lyrics are disabled.')

    job = jobs.create(filename=source.name, params=request.model_dump())

    def work(_job, report):
        return pipeline.run(
            TranscriptionRequest(
                input_path=source,
                track_name=request.track_name,
                artist_name=request.artist_name,
                lyrics_mode=mode,
                separation_preset=request.preset,
                transpose=request.transpose,
                confidence_threshold=request.confidence,
                vocals_only=request.vocals_only,
                voters=request.voters,
                assess_rhythm=request.assess_rhythm,
                force=request.force,
            ),
            progress=report)

    jobs.submit(job, work)
    return job.public()


@app.get('/api/jobs')
def list_jobs() -> Dict[str, Any]:
    return {'jobs': [job.public() for job in jobs.list()]}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')
    return job.public()


@app.delete('/api/jobs/{job_id}')
def cancel_job(job_id: str) -> Dict[str, Any]:
    if not jobs.cancel(job_id):
        raise HTTPException(status_code=409,
                            detail='Job cannot be cancelled')
    return {'cancelled': job_id}


def _finished(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=409,
                            detail=f"Job is {job.status.value}, not done")
    return job


@app.get('/api/jobs/{job_id}/notes')
def job_notes(job_id: str) -> Dict[str, Any]:
    """Full note list, for the UI's piano roll and table."""
    job = _finished(job_id)
    result = job.result
    return {
        'notes': [n.to_dict() for n in result.notes],
        'key': result.key.name if result.key else None,
        'duration': result.duration,
        'lyrics': result.lyrics.summary() if result.lyrics else None,
        'rhythm': result.rhythm.to_dict() if result.rhythm else None,
        'warnings': result.warnings,
    }


@app.get('/api/jobs/{job_id}/download/{fmt}')
def download(job_id: str, fmt: str):
    """Render the transcription in the requested format."""
    job = _finished(job_id)
    notes = job.result.notes
    title = job.params.get('track_name', '') or 'meloscribe'
    artist = job.params.get('artist_name', '')

    if fmt == 'midi':
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = EXPORT_DIR / f"{job_id}.mid"
        try:
            write_midi(notes, path)
        except ImportError as exc:
            raise HTTPException(status_code=501, detail=str(exc))
        return FileResponse(path, filename=f"{title}.mid",
                            media_type='audio/midi')

    renderers = {
        'table': lambda: format_table(notes),
        'csv': lambda: format_csv(notes),
        'json': lambda: format_json(notes),
        'leadsheet': lambda: format_leadsheet(notes),
        'lrc': lambda: format_lrc(notes, title=title, artist=artist),
    }
    if fmt not in renderers:
        raise HTTPException(status_code=400, detail=f"Unknown format {fmt!r}")

    media = {'csv': 'text/csv', 'json': 'application/json'}.get(fmt, 'text/plain')
    extension = {'leadsheet': 'txt', 'table': 'txt'}.get(fmt, fmt)

    return PlainTextResponse(
        renderers[fmt](), media_type=media,
        headers={'Content-Disposition':
                 f'attachment; filename="{title}.{extension}"'})


@app.on_event('shutdown')
def _shutdown() -> None:
    jobs.shutdown()
