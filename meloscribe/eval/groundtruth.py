"""Ground-truth and prediction containers, plus conversions between the two
representations every melody metric needs: a frame-level f0 series and a list
of discrete notes.

Frame series use 0.0 Hz to mean "unvoiced", matching the mir_eval convention.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 10ms is the CREPE/mir_eval convention; everything is resampled onto this grid
# before scoring so systems with different hop sizes stay comparable.
DEFAULT_HOP = 0.01


@dataclass
class Note:
    """A discrete note: MIDI pitch held over a half-open time interval."""
    onset: float
    offset: float
    midi: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class GroundTruth:
    """Reference annotation for one audio file."""
    name: str
    audio_path: Optional[Path] = None
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    freqs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    notes: Optional[List[Note]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.freqs = np.asarray(self.freqs, dtype=float)
        if self.times.shape != self.freqs.shape:
            raise ValueError(
                f"{self.name}: times/freqs length mismatch "
                f"({self.times.shape} vs {self.freqs.shape})"
            )
        if self.audio_path is not None:
            self.audio_path = Path(self.audio_path)

    @property
    def voiced_fraction(self) -> float:
        if self.times.size == 0:
            return 0.0
        return float(np.mean(self.freqs > 0))


@dataclass
class Prediction:
    """What a transcription system produced for one audio file."""
    name: str
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    freqs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    notes: Optional[List[Note]] = None
    runtime_s: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.freqs = np.asarray(self.freqs, dtype=float)


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0))


def hz_to_midi(hz):
    hz = np.asarray(hz, dtype=float)
    out = np.zeros_like(hz)
    positive = hz > 0
    out[positive] = 69.0 + 12.0 * np.log2(hz[positive] / 440.0)
    return out


def notes_to_f0(notes: Sequence[Note],
                hop: float = DEFAULT_HOP,
                duration: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterise discrete notes onto a uniform frame grid.

    Where notes overlap the later-starting note wins, which matches how a
    monophonic melody line is heard. Gaps between notes become unvoiced (0 Hz).
    """
    if not notes:
        n = 0 if duration is None else int(np.ceil(duration / hop))
        return np.arange(n) * hop, np.zeros(n)

    if duration is None:
        duration = max(n.offset for n in notes)

    n_frames = int(np.ceil(duration / hop))
    times = np.arange(n_frames) * hop
    freqs = np.zeros(n_frames)

    for note in sorted(notes, key=lambda x: x.onset):
        start = max(0, min(int(np.ceil(note.onset / hop)), n_frames))
        end = max(0, min(int(np.ceil(note.offset / hop)), n_frames))
        if end > start:
            freqs[start:end] = midi_to_hz(note.midi)

    return times, freqs


def f0_to_notes(times: np.ndarray,
                freqs: np.ndarray,
                min_duration: float = 0.05,
                tolerance_semitones: float = 0.5) -> List[Note]:
    """Collapse a frame-level f0 series into discrete notes.

    A note continues while the frame pitch stays within `tolerance_semitones`
    of the running note. Runs shorter than `min_duration` are dropped as
    detection jitter rather than emitted as spurious grace notes.
    """
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if times.size == 0:
        return []

    midi = hz_to_midi(freqs)
    voiced = freqs > 0
    hop = float(np.median(np.diff(times))) if times.size > 1 else DEFAULT_HOP

    notes: List[Note] = []
    state: Dict[str, Any] = {'start': None, 'pitches': []}

    def close_run(end_idx: int) -> None:
        start = state['start']
        pitches = state['pitches']
        if start is None or not pitches:
            return
        onset = float(times[start])
        offset = float(times[end_idx - 1] + hop)
        if offset - onset >= min_duration:
            notes.append(Note(onset=onset, offset=offset,
                              midi=float(np.median(pitches))))

    for i in range(len(times)):
        if not voiced[i]:
            close_run(i)
            state['start'], state['pitches'] = None, []
            continue

        if state['start'] is None:
            state['start'], state['pitches'] = i, [midi[i]]
            continue

        # Compare against the run's running centre, not the previous frame, so
        # slow drift cannot ratchet a note arbitrarily far from where it began.
        centre = float(np.median(state['pitches']))
        if abs(midi[i] - centre) <= tolerance_semitones:
            state['pitches'].append(midi[i])
        else:
            close_run(i)
            state['start'], state['pitches'] = i, [midi[i]]

    close_run(len(times))
    return notes


def resample_f0(times: np.ndarray,
                freqs: np.ndarray,
                target_times: np.ndarray) -> np.ndarray:
    """Put an f0 series onto another time grid using nearest-frame lookup.

    Interpolation is deliberately avoided: averaging across an unvoiced gap or
    a note boundary invents pitches that were never estimated.
    """
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    target_times = np.asarray(target_times, dtype=float)

    if times.size == 0:
        return np.zeros_like(target_times)

    idx = np.clip(np.searchsorted(times, target_times), 0, times.size - 1)
    left = np.clip(idx - 1, 0, times.size - 1)
    take_left = np.abs(target_times - times[left]) < np.abs(target_times - times[idx])
    idx = np.where(take_left, left, idx)

    out = freqs[idx].astype(float)
    # Anything past the end of the estimate is unvoiced, not a held final note.
    hop = float(np.median(np.diff(times))) if times.size > 1 else DEFAULT_HOP
    out[target_times > times[-1] + hop] = 0.0
    return out


def load_csv_f0(path, name: Optional[str] = None,
                audio_path=None) -> GroundTruth:
    """Load a two-column `time,frequency` CSV annotation.

    This is the vocadito / MedleyDB melody annotation format, and also the
    easiest thing for a human to hand-write for a one-off test case.
    """
    path = Path(path)
    times: List[float] = []
    freqs: List[float] = []

    with open(path, 'r', encoding='utf-8', newline='') as f:
        for row in csv.reader(f):
            if not row or len(row) < 2:
                continue
            try:
                t, hz = float(row[0]), float(row[1])
            except ValueError:
                continue  # header row
            times.append(t)
            freqs.append(max(0.0, hz))

    if not times:
        raise ValueError(f"No usable time,frequency rows found in {path}")

    return GroundTruth(
        name=name or path.stem,
        audio_path=Path(audio_path) if audio_path else None,
        times=np.array(times),
        freqs=np.array(freqs),
    )
