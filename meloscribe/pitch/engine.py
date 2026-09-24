"""The pitch engine: audio in, notes with calibrated confidence out.

Orchestrates the whole chain - condition the audio, poll every available voter,
fuse and decode, then segment the frame sequence into notes. Each note carries
both an overall confidence and the per-voter breakdown behind it, so the UI can
answer "why is this note marked uncertain?" rather than just asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import audio as audio_mod
from ..audio import Audio
from ..key import SPELLINGS
from .fusion import DecodedFrame, FusionSettings, decode, key_prior_vector
from .grid import HOP, n_frames_for
from .voters import AUTO, DEFAULT_VOTERS, Voter, build_voters

ProgressFn = Callable[[float, str], None]


def midi_to_name(midi: float, names: Sequence[str] = SPELLINGS['sharp']) -> str:
    """`names` is what each pitch class is called, index = pitch class - a
    key's `pitch_names`."""
    rounded = int(round(midi))
    name = names[rounded % 12]
    # The octave number goes with the letter, not the sounding pitch: C#
    # minor's leading tone at MIDI 60 is B#3, the same pitch as C4.
    alter = name.count('#') - name.count('b')
    return f"{name}{(rounded - alter) // 12 - 1}"


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
    # Which aligned word the note sings, as an index into the song's words;
    # None without word-level alignment. `lyric` alone cannot tell a word
    # held over several notes from the same word sung again ("na na na").
    word: Optional[int] = None
    # Filled in by the rhythm stage when a trustworthy beat grid was found;
    # None means "not assessed", which is not the same as "on the beat".
    beat_deviation: Optional[float] = None   # beats from the nearest grid slot
    duration_beats: Optional[float] = None
    # What each pitch class is called, set by the pipeline from the key the
    # notes are written in (see key.note_names). One table shared by every
    # note, and left out of to_dict() and repr, where `name` already says
    # what it decided. It changes the name only: `midi`, and with it the MIDI
    # export, is the same whatever the table.
    pitch_names: Tuple[str, ...] = field(default=SPELLINGS['sharp'], repr=False)

    @property
    def name(self) -> str:
        return midi_to_name(self.midi, self.pitch_names)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def transposed(self, semitones: int) -> 'TranscribedNote':
        note = TranscribedNote(**{**self.__dict__})
        note.midi = self.midi + semitones
        return note

    def to_dict(self) -> Dict:
        rhythm = {}
        if self.beat_deviation is not None:
            rhythm = {'beat_deviation': round(self.beat_deviation, 3),
                      'duration_beats': round(self.duration_beats or 0.0, 3)}
        return {
            **rhythm,
            'note': self.name,
            'midi': self.midi,
            'start_time': round(self.start, 3),
            'end_time': round(self.end, 3),
            'duration': round(self.duration, 3),
            'confidence': round(self.confidence, 4),
            'cents_off': round(self.pitch_cents, 1),
            'voters': {k: round(v, 3) for k, v in self.voter_scores.items()},
            'lyric': self.lyric,
            **({'word': self.word} if self.word is not None else {}),
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
    # A same-pitch split also requires the level to have climbed this far out
    # of its recent trough, as a fraction of the current level. Vibrato peaks
    # the onset detector without a real attack, so onset evidence alone is not
    # sufficient to split a note.
    #
    # Swept 0.12-0.55 against the note-F1 of every case that can express
    # over- or under-splitting. 0.45 is the middle of a plateau where all of
    # them score 1.000: below it soft-attack material shreds (0.80 at 0.12),
    # above it genuine re-articulations start being missed (0.92 at 0.55).
    # The old 0.12 was tuned against the previous, scale-dependent envelope
    # and does not mean the same thing here.
    #
    # Re-swept with the fresh-climb gate in `_has_attack`: every core case
    # stays 1.000 from 0.35 to 0.45. 0.40 bought one more split on the
    # stress suite (0.892 -> 0.898) but cost a spurious split on real solo
    # singing (vocadito_1 note F1 0.583 -> 0.574), so it stays at 0.45.
    attack_threshold: float = 0.45
    # HPSS denoising is off by default: it measured neutral-to-harmful on the
    # synthetic set. That set has no percussive bleed though - which is the
    # only thing HPSS removes - so this must be re-tested on real stems
    # before the default is treated as settled.
    #
    # Re-tested on synthetic drum bleed (`runner --suite hard`, system
    # ensemble_hpss): still harmful. Stress VFA 0.191 -> 0.352 and onset
    # F1@25 0.811 -> 0.523: the harmonic part is median-filtered over ~1s, so
    # notes smear into the rests and onsets move early. It also never reaches
    # basic-pitch, which reads the original file, so the onset matrix sees
    # the drums either way. The drum cases score note F1 1.000 without it
    # once the attack gate checks for a fresh climb (see `_has_attack`).
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
        # Voters that read from disk (basic-pitch) need a file, so the
        # conditioned array carries its provenance along with it: the
        # original, unless loading had to repair it (NaN samples, cancelling
        # channels) - then they get the repaired audio, or they would crash
        # or hear silence.
        prepared.path = audio_mod.usable_path(prepared, audio_path)

        # A clip shorter than the shortest note we emit cannot contain one,
        # and basic-pitch's note decoding crashes outright ("zero-size array")
        # on a clip shorter than one of its ~12ms frames.
        if prepared.duration < self.settings.min_note_duration:
            empty = np.zeros(0)
            return TranscriptionResult(
                notes=[], frames=DecodedFrame(times=empty, midi=empty,
                                              confidence=empty,
                                              voiced=empty.astype(bool)),
                duration=prepared.duration, voters_used=[])

        n_frames = n_frames_for(prepared.duration)

        outputs = []
        voters = self.voters
        for i, voter in enumerate(voters):
            if progress:
                progress(0.05 + 0.7 * i / len(voters), f"voter: {voter.name}")
            outputs.append(voter.observe(prepared, n_frames))

        if progress:
            progress(0.8, 'decoding')

        onsets = self._onset_matrix(outputs, n_frames)
        attacks = self._attack_envelope(prepared, n_frames)
        prior = key_prior_vector(key_pitch_classes) if key_pitch_classes else None
        frames = decode(outputs, self.settings.fusion, prior,
                        reattacks=self._reattacks(onsets, attacks))

        if progress:
            progress(0.9, 'segmenting notes')

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

    # How far back to look for the trough an attack rises out of. 80ms is long
    # enough to contain a real re-articulation's dip and short enough not to
    # reach back into the previous note.
    ATTACK_WINDOW_S = 0.08

    @classmethod
    def _attack_envelope(cls, audio: Audio, n_frames: int) -> np.ndarray:
        """Per-frame attack strength: how far the level has climbed out of its
        recent trough, as a fraction of the current level.

        This is what separates a re-articulated note from vibrato. Both produce
        peaks in the onset activation, but only a real attack is preceded by a
        dip in loudness - vibrato modulates frequency at roughly constant
        level. Without this gate a 5Hz vibrato split held notes into 200ms
        fragments at exactly the vibrato period.

        The measure is *relative* on purpose. The first version divided the
        frame-to-frame rise by the 99th percentile of rises across the whole
        track, which silently assumed the track contains some hard attacks to
        set the scale by. On softly-sung material there are none, the scale
        collapses to the size of the vibrato ripple, and the ripple clears the
        threshold: measured note precision 0.57 on a soft-attack phrase, with a
        one-second closing note shredded into five pieces. Dividing by the
        local level instead makes the number scale-free, so the same threshold
        means the same thing whether the singer punches or breathes.
        """
        from .grid import HOP as GRID_HOP
        from .grid import resample_series

        times, rms = audio_mod.rms_envelope(audio, hop=GRID_HOP)
        if rms.size == 0:
            return np.zeros(n_frames, dtype=np.float32)

        span = max(1, int(round(cls.ATTACK_WINDOW_S / GRID_HOP)))
        # Trough of the preceding window, inclusive of the current frame.
        padded = np.pad(rms, (span, 0), mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(padded, span + 1)
        trough = windows.min(axis=1)[:rms.size]

        rise = (rms - trough) / np.maximum(rms, 1e-6)
        np.clip(rise, 0.0, 1.0, out=rise)

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

    # The RMS frames are 40ms (4 hops) long, so the level finishes climbing up
    # to 40ms after basic-pitch's onset peak. Looking only 20ms ahead missed
    # re-articulations whose dip was part-filled by bleed: measured on the
    # stress suite, a split landed 70ms late in repeats_drums/repeats_harmony
    # (attack 0.44 inside the old window, 0.47 just after it).
    ATTACK_LAG_FRAMES = 4

    def _has_attack(self, attacks: Optional[np.ndarray], i: int) -> bool:
        """Whether the amplitude envelope genuinely climbs at frame i.

        Checked over a window rather than the single frame, since the onset
        activation peak and the loudness climb need not land on the same 10ms
        frame. The climb is measured against the envelope's lowest point over
        the preceding ATTACK_WINDOW_S, not against zero: a slow attack keeps
        the envelope high for ~80ms after it ends, and that stale tail was
        validating a second onset peak 150ms into a soft note (scale_drums_loud
        split a held note in two). Measured with both changes: stress-suite
        note F1 0.864 -> 0.892, every core case unchanged, and one spurious
        split fewer on real solo singing (vocadito_1 0.579 -> 0.583).
        """
        if attacks is None or i >= len(attacks):
            return True  # no envelope available: fall back to onsets alone
        lo = max(0, i - 2)
        window = attacks[lo:i + self.ATTACK_LAG_FRAMES + 1]
        if not window.size:
            return False
        span = max(1, int(round(self.ATTACK_WINDOW_S / HOP)))
        before = attacks[max(0, lo - span):lo]
        base = float(before.min()) if before.size else 0.0
        return bool(window.max() - base >= self.settings.attack_threshold)

    def _reattacks(self, onsets: Optional[np.ndarray],
                   attacks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Onset peaks that come with a genuine climb in level: new notes.

        The decoder's backing-voice prior needs to know when a pitch was
        freshly attacked, and an onset peak alone cannot say: basic-pitch also
        fires on a backing voice the moment the lead releases and *unmasks*
        it, and that is exactly when the level is falling, not climbing.
        Measured: with ungated onsets the prior's evidence was reset right
        before every backing-voice tail, and it did nothing (stress VFA 0.191
        -> 0.190).
        """
        if onsets is None or attacks is None:
            return None
        climbs = np.array([self._has_attack(attacks, t)
                           for t in range(onsets.shape[0])])
        return onsets & climbs[:, None]

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
