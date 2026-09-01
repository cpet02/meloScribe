"""The pitch engine: audio in, notes with calibrated confidence out.

Orchestrates the whole chain - condition the audio, poll every available voter,
fuse and decode, then segment the frame sequence into notes. Each note carries
both an overall confidence and the per-voter breakdown behind it, so the UI can
answer "why is this note marked uncertain?" rather than just asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .. import audio as audio_mod
from ..audio import Audio
from .fusion import DecodedFrame, FusionSettings, decode, key_prior_vector
from .grid import HOP, n_frames_for
from .voters import AUTO, DEFAULT_VOTERS, Voter, build_voters

ProgressFn = Callable[[float, str], None]

NOTE_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')


def midi_to_name(midi: float) -> str:
    rounded = int(round(midi))
    return f"{NOTE_NAMES[rounded % 12]}{rounded // 12 - 1}"


@dataclass
class TranscribedNote:
    """One note, with the evidence that produced it."""
    midi: int
    start: float
    end: float
    confidence: float
    pitch_cents: float = 0.0        # deviation from equal temperament
    voter_scores: Dict[str, float] = field(default_factory=dict)
    lyric: Optional[str] = None
    syllable: Optional[str] = None

    @property
    def name(self) -> str:
        return midi_to_name(self.midi)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def transposed(self, semitones: int) -> 'TranscribedNote':
        note = TranscribedNote(**{**self.__dict__})
        note.midi = self.midi + semitones
        return note

    def to_dict(self) -> Dict:
        return {
            'note': self.name,
            'midi': self.midi,
            'start_time': round(self.start, 3),
            'end_time': round(self.end, 3),
            'duration': round(self.duration, 3),
            'confidence': round(self.confidence, 4),
            'cents_off': round(self.pitch_cents, 1),
            'voters': {k: round(v, 3) for k, v in self.voter_scores.items()},
            'lyric': self.lyric,
        }


@dataclass
class TranscriptionResult:
    notes: List[TranscribedNote]
    frames: DecodedFrame
    duration: float
    voters_used: List[str]
    settings: Dict = field(default_factory=dict)

    @property
    def mean_confidence(self) -> float:
        return float(np.mean([n.confidence for n in self.notes])) if self.notes else 0.0

    def filter_confidence(self, threshold: float) -> List[TranscribedNote]:
        return [n for n in self.notes if n.confidence >= threshold]


@dataclass
class EngineSettings:
    # 'auto' resolves by what is installed - see voters.DEFAULT_VOTERS.
    voters: Sequence[str] = AUTO
    fusion: FusionSettings = field(default_factory=FusionSettings)
    min_note_duration: float = 0.06
    # A same-pitch split also requires the amplitude envelope to rise by this
    # much (normalised). Vibrato peaks the onset detector without a real
    # attack, so onset evidence alone is not sufficient to split a note.
    attack_threshold: float = 0.12
    # HPSS denoising is off by default: it measured neutral-to-harmful on the
    # synthetic set. That set has no percussive bleed though - which is the
    # only thing HPSS removes - so this must be re-tested on real stems
    # before the default is treated as settled.
    denoise: bool = False
    # Notes below this are kept but flagged, never silently dropped - the UI
    # shows them greyed so a wrong call is visible rather than invisible.
    low_confidence: float = 0.5


class PitchEngine:
    """Multi-voter melody transcription."""

    def __init__(self, settings: Optional[EngineSettings] = None):
        self.settings = settings or EngineSettings()
        self._voters: Optional[List[Voter]] = None

    @property
    def voters(self) -> List[Voter]:
        if self._voters is None:
            self._voters = build_voters(self.settings.voters)
        return self._voters

    def transcribe(self, audio_path, key_pitch_classes: Optional[Sequence[int]] = None,
                   progress: Optional[ProgressFn] = None) -> TranscriptionResult:
        audio_path = Path(audio_path)
        if progress:
            progress(0.0, 'conditioning audio')

        prepared = audio_mod.prepare_for_pitch(audio_path,
                                               denoise=self.settings.denoise)
        # Voters that read from disk (basic-pitch) need the original file, so
        # the conditioned array carries its provenance along with it.
        prepared.path = audio_path
        n_frames = n_frames_for(prepared.duration)

        outputs = []
        voters = self.voters
        for i, voter in enumerate(voters):
            if progress:
                progress(0.05 + 0.7 * i / len(voters), f"voter: {voter.name}")
            outputs.append(voter.observe(prepared, n_frames))

        if progress:
            progress(0.8, 'decoding')

        prior = key_prior_vector(key_pitch_classes) if key_pitch_classes else None
        frames = decode(outputs, self.settings.fusion, prior)

        if progress:
            progress(0.9, 'segmenting notes')

        onsets = self._onset_matrix(outputs, n_frames)
        attacks = self._attack_envelope(prepared, n_frames)
        notes = self._segment(frames, onsets, attacks)

        if progress:
            progress(1.0, f"{len(notes)} notes")

        return TranscriptionResult(
            notes=notes,
            frames=frames,
            duration=prepared.duration,
            voters_used=[v.name for v in voters],
            settings={'voters': [v.name for v in voters],
                      'fusion': self.settings.fusion.__dict__},
        )

    @staticmethod
    def _onset_matrix(outputs, n_frames: int) -> Optional[np.ndarray]:
        """basic-pitch's onset activations, reduced to genuine attack peaks.

        Needed to split *repeated* notes: two quarter-notes on the same pitch
        look like one half-note to a pitch tracker, and only an onset detector
        can tell them apart.

        The raw activation cannot be thresholded directly. It stays elevated
        for the whole duration of a sung note and ripples with vibrato, so a
        bare `>= threshold` test splits every held note into a stutter of
        fragments - measured at note F1 0.00 on sustained material, against
        0.31 for not splitting at all. What identifies a real attack is a
        local *peak*, so the activation is peak-picked and each peak must also
        clear the local baseline.
        """
        for out in outputs:
            if 'onsets' not in out.meta:
                continue

            onsets = np.asarray(out.meta['onsets'], dtype=np.float32)
            if onsets.shape[0] < 3:
                return None

            # A peak must exceed both neighbours and stand clear of the
            # surrounding activation, not merely be locally largest - vibrato
            # produces local maxima too, just shallow ones.
            previous = np.vstack([onsets[:1], onsets[:-1]])
            following = np.vstack([onsets[1:], onsets[-1:]])
            is_peak = (onsets > previous) & (onsets >= following)
            prominent = onsets > (np.median(onsets, axis=0, keepdims=True) + 0.25)

            return (is_peak & prominent & (onsets >= 0.5))
        return None

    @staticmethod
    def _attack_envelope(audio: Audio, n_frames: int) -> np.ndarray:
        """Per-frame amplitude attack strength, normalised to [0, 1].

        This is what separates a re-articulated note from vibrato. Both produce
        peaks in the onset activation, but only a real attack raises the
        amplitude envelope - vibrato modulates frequency at roughly constant
        loudness. Requiring a rise in loudness at the split point is therefore
        the discriminator; without it, a 5Hz vibrato was splitting held notes
        into 200ms fragments at exactly the vibrato period.
        """
        from .grid import HOP as GRID_HOP
        from .grid import resample_series

        times, rms = audio_mod.rms_envelope(audio, hop=GRID_HOP)
        rise = np.diff(rms, prepend=rms[:1])
        np.maximum(rise, 0.0, out=rise)

        peak = float(np.percentile(rise, 99)) if rise.size else 0.0
        if peak > 0:
            rise = np.clip(rise / peak, 0.0, 1.0)

        return resample_series(rise, times, n_frames)

    def _segment(self, frames: DecodedFrame,
                 onsets: Optional[np.ndarray] = None,
                 attacks: Optional[np.ndarray] = None) -> List[TranscribedNote]:
        """Group decoded frames into notes."""
        notes: List[TranscribedNote] = []
        n_frames = len(frames.times)
        if n_frames == 0:
            return notes

        rounded = np.where(frames.voiced, np.round(frames.midi), -1)

        # A re-articulation cannot legally follow the previous attack sooner
        # than this. Without a refractory window a single messy attack fires
        # several adjacent peaks and splits the note it just started.
        refractory = max(self.settings.min_note_duration, 0.12)

        start = 0
        for i in range(1, n_frames + 1):
            elapsed = (i - start) * HOP
            ends = (i == n_frames
                    or rounded[i] != rounded[start]
                    or (elapsed >= refractory
                        and self._is_reonset(onsets, rounded, i)
                        and self._has_attack(attacks, i)))
            if not ends:
                continue

            if rounded[start] >= 0:
                note = self._make_note(frames, start, i)
                if note is not None:
                    notes.append(note)
            start = i

        return notes

    def _has_attack(self, attacks: Optional[np.ndarray], i: int) -> bool:
        """Whether the amplitude envelope actually rises at frame i.

        Checked over a short window rather than the single frame, since the
        onset activation peak and the loudness peak need not land on the same
        10ms frame.
        """
        if attacks is None or i >= len(attacks):
            return True  # no envelope available: fall back to onsets alone
        window = attacks[max(0, i - 2):i + 3]
        return bool(window.size and window.max() >= self.settings.attack_threshold)

    @staticmethod
    def _is_reonset(onsets: Optional[np.ndarray], rounded: np.ndarray,
                    i: int) -> bool:
        """Whether frame i re-articulates the note already sounding.

        `onsets` is the peak-picked boolean matrix from `_onset_matrix`, so
        this is a lookup rather than a threshold test.
        """
        if onsets is None or i >= onsets.shape[0] or rounded[i] < 0:
            return False
        from .grid import MIDI_MIN
        col = int(rounded[i]) - MIDI_MIN
        if not 0 <= col < onsets.shape[1]:
            return False
        return bool(onsets[i, col])

    def _make_note(self, frames: DecodedFrame, start: int,
                   end: int) -> Optional[TranscribedNote]:
        duration = (end - start) * HOP
        if duration < self.settings.min_note_duration:
            return None

        span = slice(start, end)
        midi_values = frames.midi[span]
        confidence = float(np.mean(frames.confidence[span]))

        # Cents deviation is measured from the median rather than the mean, so
        # a scooped attack or a vibrato peak cannot drag the reported
        # intonation away from where the note actually sat.
        centre = float(np.median(midi_values))
        cents = (centre - round(centre)) * 100.0

        return TranscribedNote(
            midi=int(round(centre)),
            start=float(frames.times[start]),
            end=float(frames.times[start] + duration),
            confidence=confidence,
            pitch_cents=cents,
            voter_scores={name: float(np.mean(values[span]))
                          for name, values in frames.per_voter.items()},
        )


def transcribe(audio_path, voters: Sequence[str] = DEFAULT_VOTERS,
               progress: Optional[ProgressFn] = None,
               **kwargs) -> TranscriptionResult:
    """Convenience entry point."""
    return PitchEngine(EngineSettings(voters=voters, **kwargs)).transcribe(
        audio_path, progress=progress)
