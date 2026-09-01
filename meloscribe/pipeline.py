"""Stage-based orchestration.

The old `main.py` was a straight-line script: any change re-ran everything from
separation onward, and the only progress signal was whatever the libraries
happened to print. Here the work is a list of named stages with declared
weights, so the caller - CLI or web UI - gets meaningful progress, and each
stage's expensive output is content-addressed and reused.

Stage order matters in one specific way: the track name is validated *before*
separation. Learning that a required field was missing after a five-minute
separation would be indefensible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .key import KeyEstimate, estimate_key
from .lyrics.lrclib import TrackQuery, describe_track
from .lyrics.service import LyricsMode, LyricsOutcome, LyricsService
from .pitch.engine import (EngineSettings, PitchEngine, TranscribedNote,
                           TranscriptionResult)
from .stems import Separator, SeparationSettings, StemResult, best_device

ProgressFn = Callable[[float, str], None]

# Relative cost of each stage, for a progress bar that does not lie. Separation
# genuinely dominates, so pretending the stages are equal would park the bar at
# 20% for most of the run.
STAGE_WEIGHTS = {
    'metadata': 0.02,
    'separate': 0.55,
    'key': 0.03,
    'transcribe': 0.30,
    'lyrics': 0.08,
    'output': 0.02,
}


@dataclass
class TranscriptionRequest:
    """Everything needed to run the pipeline once."""
    input_path: Path
    track_name: str = ''
    artist_name: str = ''
    lyrics_mode: LyricsMode = LyricsMode.ALIGN
    separation_preset: str = 'balanced'
    voters: Optional[Sequence[str]] = None
    transpose: int = 0
    confidence_threshold: float = 0.0
    device: Optional[str] = None
    use_key_prior: bool = True
    force: bool = False
    # Skip separation when the input is already an isolated vocal.
    vocals_only: bool = False

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        if isinstance(self.lyrics_mode, str):
            self.lyrics_mode = LyricsMode(self.lyrics_mode)

    def as_query(self, duration: Optional[float] = None) -> TrackQuery:
        return TrackQuery(track_name=self.track_name.strip(),
                          artist_name=self.artist_name.strip(),
                          duration=duration)


@dataclass
class TranscriptionOutput:
    """The finished product."""
    notes: List[TranscribedNote]
    key: Optional[KeyEstimate] = None
    lyrics: Optional[LyricsOutcome] = None
    stems: Optional[StemResult] = None
    transcription: Optional[TranscriptionResult] = None
    duration: float = 0.0
    elapsed_s: float = 0.0
    request: Optional[TranscriptionRequest] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        if not self.notes:
            return 0.0
        return sum(n.confidence for n in self.notes) / len(self.notes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'notes': [n.to_dict() for n in self.notes],
            'key': self.key.name if self.key else None,
            'key_confidence': round(self.key.confidence, 3) if self.key else None,
            'lyrics': self.lyrics.summary() if self.lyrics else None,
            'duration': round(self.duration, 2),
            'elapsed_s': round(self.elapsed_s, 2),
            'mean_confidence': round(self.mean_confidence, 3),
            'note_count': len(self.notes),
            'warnings': self.warnings,
        }


class _Progress:
    """Maps per-stage progress onto one overall 0-1 bar."""

    def __init__(self, callback: Optional[ProgressFn]):
        self.callback = callback
        self.completed = 0.0
        self.current = 0.0

    def stage(self, name: str):
        self.current = STAGE_WEIGHTS.get(name, 0.05)

        def report(fraction: float, message: str) -> None:
            if self.callback:
                overall = self.completed + self.current * max(0.0, min(1.0, fraction))
                self.callback(min(overall, 1.0), message)

        return report

    def finish_stage(self) -> None:
        self.completed = min(1.0, self.completed + self.current)


class Pipeline:
    """Runs the full transcription workflow."""

    def __init__(self, lyrics_service: Optional[LyricsService] = None):
        self.lyrics_service = lyrics_service or LyricsService()

    def run(self, request: TranscriptionRequest,
            progress: Optional[ProgressFn] = None) -> TranscriptionOutput:
        started = time.perf_counter()
        tracker = _Progress(progress)
        output = TranscriptionOutput(notes=[], request=request)

        if not request.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {request.input_path}")

        # --- metadata + the naming gate ------------------------------------
        report = tracker.stage('metadata')
        report(0.2, 'reading track metadata')

        query = request.as_query()
        if not query.is_usable() and request.lyrics_mode != LyricsMode.OFF:
            # Fall back to tags/filename before refusing outright - the user
            # should only be stopped when we genuinely cannot work it out.
            guessed = describe_track(request.input_path)
            query.track_name = query.track_name or guessed.track_name
            query.artist_name = query.artist_name or guessed.artist_name
            if guessed.track_name:
                output.warnings.append(
                    f"No track name given; using '{guessed.track_name}' from "
                    f"the file's metadata.")

        # Raises MissingTrackName, before any expensive stage runs.
        LyricsService.require_track_name(query, request.lyrics_mode)
        tracker.finish_stage()

        # --- separation ----------------------------------------------------
        if request.vocals_only:
            vocals_path = request.input_path
            tracker.current = STAGE_WEIGHTS['separate']
            tracker.finish_stage()
        else:
            report = tracker.stage('separate')
            device = best_device(request.device)
            separator = Separator(
                settings=SeparationSettings.preset(request.separation_preset, device),
                device=device)
            output.stems = separator.separate(request.input_path,
                                              progress=report,
                                              force=request.force)
            vocals_path = output.stems.vocals
            tracker.finish_stage()

        # --- key estimation ------------------------------------------------
        key_prior = None
        if request.use_key_prior:
            report = tracker.stage('key')
            report(0.3, 'estimating key')
            try:
                output.key = estimate_key(request.input_path)
                key_prior = output.key.as_prior()
                if key_prior is None:
                    output.warnings.append(
                        f"Key estimate ({output.key.name}) too uncertain to use "
                        f"as a prior; transcribing without it.")
            except Exception as exc:
                output.warnings.append(f"Key estimation failed: {exc}")
            tracker.finish_stage()

        # --- transcription -------------------------------------------------
        report = tracker.stage('transcribe')
        settings = EngineSettings(voters=request.voters) if request.voters \
            else EngineSettings()
        result = PitchEngine(settings).transcribe(
            vocals_path, key_pitch_classes=key_prior, progress=report)
        output.transcription = result
        output.duration = result.duration
        tracker.finish_stage()

        notes = result.notes
        if request.confidence_threshold > 0:
            kept = [n for n in notes if n.confidence >= request.confidence_threshold]
            if len(kept) < len(notes):
                output.warnings.append(
                    f"Confidence filter removed {len(notes) - len(kept)} of "
                    f"{len(notes)} notes.")
            notes = kept

        # --- lyrics --------------------------------------------------------
        report = tracker.stage('lyrics')
        query.duration = query.duration or result.duration
        output.lyrics = self.lyrics_service.resolve(
            vocals_path, query, mode=request.lyrics_mode,
            duration=result.duration, progress=report)
        output.warnings.extend(output.lyrics.warnings)

        from .lyrics.align import attach_to_notes
        attach_to_notes(notes, output.lyrics.lyrics if output.lyrics.found else None)
        tracker.finish_stage()

        # --- output --------------------------------------------------------
        report = tracker.stage('output')
        if request.transpose:
            notes = [n.transposed(request.transpose) for n in notes]
        output.notes = notes
        report(1.0, f"{len(notes)} notes")
        tracker.finish_stage()

        output.elapsed_s = time.perf_counter() - started
        if progress:
            progress(1.0, 'complete')
        return output


def transcribe_file(input_path, track_name: str = '', artist_name: str = '',
                    progress: Optional[ProgressFn] = None,
                    **kwargs) -> TranscriptionOutput:
    """Convenience entry point for a single file."""
    request = TranscriptionRequest(input_path=input_path, track_name=track_name,
                                   artist_name=artist_name, **kwargs)
    return Pipeline().run(request, progress=progress)
