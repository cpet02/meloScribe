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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .key import (SPELLINGS, KeyEstimate, estimate_key, note_names,
                  note_spelling)
from .lyrics.lrclib import TrackQuery, describe_track
from .lyrics.service import LyricsMode, LyricsOutcome, LyricsService
from .pitch.engine import (EngineSettings, PitchEngine, TranscribedNote,
                           TranscriptionResult)
from .rhythm import RhythmReport
from .stems import Separator, SeparationSettings, StemResult, best_device

ProgressFn = Callable[[float, str], None]

# Relative cost of each stage, for a progress bar that does not lie. Separation
# genuinely dominates, so pretending the stages are equal would park the bar at
# 20% for most of the run.
STAGE_WEIGHTS = {
    'metadata': 0.02,
    'separate': 0.52,
    'key': 0.03,
    'transcribe': 0.29,
    'lyrics': 0.08,
    'rhythm': 0.04,
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
    # Rhythmic plausibility is reported, never acted on: it flags a
    # transcription for review and does not change a single note.
    #
    # Off by default. It is a diagnostic whose threshold rests on a single real
    # track, and until that evidence is broader it should not add four seconds
    # and two columns to every run that did not ask for it. Opt in with
    # `--rhythm`, or `assess_rhythm=True`.
    assess_rhythm: bool = False
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
    rhythm: Optional['RhythmReport'] = None
    duration: float = 0.0
    elapsed_s: float = 0.0
    request: Optional[TranscriptionRequest] = None
    warnings: List[str] = field(default_factory=list)
    # How the notes were named: the written key's accidentals, 'sharp' or
    # 'flat', and its name for each pitch class, index = pitch class.
    # Recorded rather than recomputed, so what is reported always matches
    # the names.
    spelling: str = 'sharp'
    pitch_names: Tuple[str, ...] = SPELLINGS['sharp']

    @property
    def mean_confidence(self) -> float:
        if not self.notes:
            return 0.0
        return sum(n.confidence for n in self.notes) / len(self.notes)

    @property
    def written_key(self) -> Optional[KeyEstimate]:
        """The key the transposed notes are written in; None when they are
        not transposed, and `key` - the concert key - already is it."""
        transpose = self.request.transpose if self.request else 0
        return self.key.transposed(transpose) if self.key and transpose else None

    def to_dict(self) -> Dict[str, Any]:
        written = self.written_key
        return {
            'notes': [n.to_dict() for n in self.notes],
            'key': self.key.name if self.key else None,
            'written_key': written.name if written else None,
            'key_confidence': round(self.key.confidence, 3) if self.key else None,
            'lyrics': self.lyrics.summary() if self.lyrics else None,
            'rhythm': self.rhythm.to_dict() if self.rhythm else None,
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
                        f"as a prior; transcribing without it, and spelling "
                        f"notes with sharps.")
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

        # --- rhythmic plausibility -----------------------------------------
        # Deliberately last, and deliberately read-only. Nothing downstream
        # consumes it: it annotates the notes and produces a flag, and the
        # note list that comes out is byte-identical to the one that went in.
        if request.assess_rhythm:
            report = tracker.stage('rhythm')
            report(0.2, 'tracking beats')
            try:
                output.rhythm = self._assess_rhythm(request, notes)
            except Exception as exc:
                output.warnings.append(f"Rhythm analysis failed: {exc}")
            tracker.finish_stage()
        else:
            # Retire the stage's weight anyway, or the bar stops at 96%.
            tracker.current = STAGE_WEIGHTS['rhythm']
            tracker.finish_stage()

        # --- output --------------------------------------------------------
        report = tracker.stage('output')
        if request.transpose:
            notes = [n.transposed(request.transpose) for n in notes]
        # Spelled for the key the player reads - the written key, once
        # transposed - so the names agree with its key signature.
        output.spelling = note_spelling(output.key, request.transpose)
        output.pitch_names = note_names(output.key, request.transpose)
        for note in notes:
            note.pitch_names = output.pitch_names
        output.notes = notes
        report(1.0, f"{len(notes)} notes")
        tracker.finish_stage()

        output.elapsed_s = time.perf_counter() - started
        if progress:
            progress(1.0, 'complete')
        return output


    @staticmethod
    def _assess_rhythm(request: TranscriptionRequest,
                       notes: List[TranscribedNote]) -> RhythmReport:
        """Score the notes against a beat grid tracked from the full mix.

        The *mix*, not the vocal stem: beat tracking keys off percussive
        onsets, which separation has deliberately removed. A grid tracked from
        the vocals would be derived from the singer's own phrasing and would
        then be used to judge that same phrasing, which measures nothing.
        """
        from . import rhythm as rhythm_mod

        report = rhythm_mod.assess(notes, request.input_path,
                                   force=request.force)
        if request.vocals_only:
            report.warnings.append(
                'Beat tracking ran on an isolated vocal, which has no '
                'percussion to lock onto; treat the rhythm score as weak.')

        for note, detail in zip(notes, report.notes):
            note.beat_deviation = detail.deviation_beats
            note.duration_beats = detail.duration_beats

        return report


def transcribe_file(input_path, track_name: str = '', artist_name: str = '',
                    progress: Optional[ProgressFn] = None,
                    **kwargs) -> TranscriptionOutput:
    """Convenience entry point for a single file."""
    request = TranscriptionRequest(input_path=input_path, track_name=track_name,
                                   artist_name=artist_name, **kwargs)
    return Pipeline().run(request, progress=progress)
