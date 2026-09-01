"""The shared time/pitch grid every voter reports onto.

Fusion is only meaningful if the voters are speaking about the same instants
and the same pitches. Rather than have each estimator impose its own hop size
and range, they all resample onto this grid, and disagreement between them then
means genuine disagreement rather than a bookkeeping artefact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

# 10ms frames: CREPE's native rate, mir_eval's convention, and fine enough to
# place an onset within a tolerance a listener would accept.
HOP = 0.01

# C2..C7. Below C2 is bass bleed, above C7 nothing sings; keeping the range
# tight removes whole classes of spurious candidate.
MIDI_MIN = 36
MIDI_MAX = 96
N_PITCHES = MIDI_MAX - MIDI_MIN + 1

PITCHES = np.arange(MIDI_MIN, MIDI_MAX + 1, dtype=float)
PITCH_HZ = 440.0 * 2.0 ** ((PITCHES - 69.0) / 12.0)


def n_frames_for(duration: float) -> int:
    return int(np.ceil(duration / HOP))


def frame_times(n_frames: int) -> np.ndarray:
    return np.arange(n_frames) * HOP


def midi_index(midi: np.ndarray) -> np.ndarray:
    """Index into the pitch grid for a MIDI number, clipped to range."""
    return np.clip(np.round(np.asarray(midi) - MIDI_MIN).astype(int),
                   0, N_PITCHES - 1)


@dataclass
class VoterOutput:
    """One voter's opinion, on the shared grid.

    `salience` is (n_frames, N_PITCHES) in [0, 1] - how strongly this voter
    believes each pitch is sounding. `voicing` is (n_frames,) in [0, 1] - how
    strongly it believes *anything* is being sung. The two are deliberately
    separate: a voter can be confident someone is singing while being unsure
    which note, and that uncertainty should survive into the fusion rather
    than being collapsed early.
    """
    name: str
    salience: np.ndarray
    voicing: np.ndarray
    weight: float = 1.0
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.salience = np.asarray(self.salience, dtype=np.float32)
        self.voicing = np.asarray(self.voicing, dtype=np.float32)
        if self.salience.ndim != 2 or self.salience.shape[1] != N_PITCHES:
            raise ValueError(
                f"{self.name}: salience must be (n_frames, {N_PITCHES}), "
                f"got {self.salience.shape}")
        if self.voicing.shape[0] != self.salience.shape[0]:
            raise ValueError(
                f"{self.name}: voicing has {self.voicing.shape[0]} frames but "
                f"salience has {self.salience.shape[0]}")

    @property
    def n_frames(self) -> int:
        return self.salience.shape[0]

    def best_pitch(self) -> np.ndarray:
        """The argmax pitch per frame, in MIDI - for diagnostics, not decoding."""
        return PITCHES[np.argmax(self.salience, axis=1)]


def resample_matrix(matrix: np.ndarray, src_times: np.ndarray,
                    n_frames: int) -> np.ndarray:
    """Put a voter's (n_src, N_PITCHES) matrix onto the shared frame grid.

    Nearest-frame, not interpolated: blending two adjacent frames across a note
    boundary would manufacture salience for a pitch neither frame supported.
    """
    target = frame_times(n_frames)
    if src_times.size == 0:
        return np.zeros((n_frames, matrix.shape[1]), dtype=np.float32)

    idx = np.clip(np.searchsorted(src_times, target), 0, src_times.size - 1)
    left = np.clip(idx - 1, 0, src_times.size - 1)
    take_left = np.abs(target - src_times[left]) < np.abs(target - src_times[idx])
    idx = np.where(take_left, left, idx)
    return matrix[idx].astype(np.float32)


def resample_series(series: np.ndarray, src_times: np.ndarray,
                    n_frames: int) -> np.ndarray:
    """Nearest-frame resampling for a 1-D per-frame series."""
    return resample_matrix(np.asarray(series).reshape(-1, 1),
                           src_times, n_frames).ravel()


def gaussian_bump(midi: np.ndarray, weight: np.ndarray,
                  width_semitones: float = 0.35) -> np.ndarray:
    """Turn per-frame point estimates into a salience matrix.

    Voters like PYIN report a single frequency, not a distribution. Spreading
    it over a narrow Gaussian expresses the real uncertainty - a reading of
    'A4 plus 20 cents' is weak evidence for A#4 too, and pretending otherwise
    makes the fusion brittle at note boundaries.
    """
    midi = np.asarray(midi, dtype=float)
    weight = np.asarray(weight, dtype=float)
    n_frames = midi.shape[0]

    salience = np.zeros((n_frames, N_PITCHES), dtype=np.float32)
    voiced = (midi >= MIDI_MIN - 1) & (midi <= MIDI_MAX + 1) & (weight > 0)
    if not np.any(voiced):
        return salience

    distance = PITCHES[None, :] - midi[voiced][:, None]
    bump = np.exp(-0.5 * (distance / width_semitones) ** 2)
    salience[voiced] = (bump * weight[voiced][:, None]).astype(np.float32)
    return salience


def normalize_rows(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Scale each frame to a max of 1.

    Voters are comparable in *shape*, not in absolute magnitude - a loud frame
    should not outvote a quiet one simply for being loud. Per-frame
    normalisation is what makes a fixed set of fusion weights meaningful.
    """
    peak = np.max(matrix, axis=1, keepdims=True)
    return (matrix / (peak + eps)).astype(np.float32)
