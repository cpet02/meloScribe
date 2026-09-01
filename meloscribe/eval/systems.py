"""Transcription systems under test, behind one interface.

Every system takes an audio path and returns a `Prediction`. That is the whole
contract - it lets the baseline we are trying to beat and the ensemble we are
building be scored by identical code, on identical input, in the same run.

Systems are registered lazily: importing this module must not import torch or
tensorflow, so that `--list` stays instant.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .groundtruth import DEFAULT_HOP, Note, Prediction, notes_to_f0


class System:
    """Base class for anything that can transcribe a melody."""

    name = 'base'
    description = ''

    def transcribe(self, audio_path: Path) -> Prediction:
        raise NotImplementedError

    def run(self, audio_path: Path) -> Prediction:
        """Transcribe and stamp the wall-clock cost onto the prediction."""
        started = time.perf_counter()
        prediction = self.transcribe(Path(audio_path))
        prediction.runtime_s = time.perf_counter() - started
        return prediction


class BasicPitchSystem(System):
    """The current pipeline's detector: basic-pitch note events.

    This is the baseline every later change is measured against. It reproduces
    what `pipeline/pitch_detector.py` does today, deliberately including its
    weakness: `amplitude` is carried through as `confidence`, though it is a
    loudness value and not a calibrated probability.
    """

    name = 'basic_pitch'
    description = 'basic-pitch note events (current pipeline baseline)'

    def __init__(self, onset_threshold: float = 0.5,
                 frame_threshold: float = 0.3,
                 minimum_note_length: float = 0.058):
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length = minimum_note_length

    def transcribe(self, audio_path: Path) -> Prediction:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict

        _, _, note_events = predict(
            audio_path=str(audio_path),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=self.minimum_note_length,
            multiple_pitch_bends=False,
            melodia_trick=True,
        )

        notes = [
            Note(onset=float(start), offset=float(end),
                 midi=float(midi), confidence=float(amp))
            for start, end, midi, amp, _ in note_events
        ]
        notes = _monophonic(notes)
        times, freqs = notes_to_f0(notes) if notes else (np.zeros(0), np.zeros(0))

        return Prediction(name=Path(audio_path).stem, times=times, freqs=freqs,
                          notes=notes, meta={'n_raw_events': len(note_events)})


class PyinSystem(System):
    """librosa PYIN, frame level.

    Runs at full resolution rather than the downsampled settings the current
    pipeline uses for cross-validation - when PYIN is being scored as a system
    in its own right, handicapping it would make the comparison meaningless.
    """

    name = 'pyin'
    description = 'librosa PYIN frame-level f0'

    def __init__(self, fmin_note: str = 'C2', fmax_note: str = 'C7',
                 hop: float = DEFAULT_HOP):
        self.fmin_note = fmin_note
        self.fmax_note = fmax_note
        self.hop = hop

    def transcribe(self, audio_path: Path) -> Prediction:
        import librosa

        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        hop_length = max(1, int(round(self.hop * sr)))

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz(self.fmin_note),
            fmax=librosa.note_to_hz(self.fmax_note),
            sr=sr,
            hop_length=hop_length,
        )

        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        freqs = np.nan_to_num(f0, nan=0.0)
        freqs[~voiced_flag] = 0.0

        return Prediction(name=Path(audio_path).stem, times=times, freqs=freqs,
                          meta={'mean_voiced_prob': float(np.mean(voiced_prob))})


class OracleSystem(System):
    """Reads the answer off the ground truth.

    Not a real system - a harness self-test. If the oracle does not score 1.0
    the bug is in the scoring code, not in the transcriber, and every other
    number in the run is meaningless until it is fixed.
    """

    name = 'oracle'
    description = 'ground truth echoed back (harness self-test)'

    def __init__(self, truths: Optional[Dict[str, object]] = None):
        self.truths = truths or {}

    def transcribe(self, audio_path: Path) -> Prediction:
        truth = self.truths.get(Path(audio_path).stem)
        if truth is None:
            raise KeyError(f"Oracle has no ground truth for {audio_path.stem}")
        return Prediction(name=truth.name, times=truth.times.copy(),
                          freqs=truth.freqs.copy(),
                          notes=list(truth.notes) if truth.notes else None)


class EnsembleSystem(System):
    """The multi-voter engine: several estimators fused and Viterbi-decoded.

    Voter set is configurable so the harness can attribute a gain to a
    specific voter rather than to "the ensemble" as an undifferentiated
    whole - the only way to know whether a component is earning its runtime.
    """

    name = 'ensemble'
    description = 'multi-voter fusion + Viterbi decode'

    def __init__(self, voters=None, **kwargs):
        from ..pitch.engine import DEFAULT_VOTERS
        self.voters = tuple(voters) if voters else DEFAULT_VOTERS
        self.kwargs = kwargs

    def transcribe(self, audio_path: Path) -> Prediction:
        from ..pitch.engine import EngineSettings, PitchEngine

        engine = PitchEngine(EngineSettings(voters=self.voters, **self.kwargs))
        result = engine.transcribe(audio_path)

        notes = [Note(onset=n.start, offset=n.end, midi=float(n.midi),
                      confidence=n.confidence) for n in result.notes]

        # Score the decoded frame sequence directly rather than re-rasterising
        # the notes: it is what the decoder actually concluded, and rounding it
        # through note segmentation first would hide segmentation errors.
        frames = result.frames
        freqs = np.where(frames.voiced,
                         440.0 * 2 ** ((frames.midi - 69) / 12), 0.0)

        return Prediction(name=Path(audio_path).stem, times=frames.times,
                          freqs=freqs, notes=notes,
                          meta={'voters': result.voters_used,
                                'mean_confidence': result.mean_confidence})


def _monophonic(notes: List[Note]) -> List[Note]:
    """Reduce polyphonic output to a single melody line.

    basic-pitch is a polyphonic transcriber, so on a vocal stem it will happily
    emit overlapping notes from reverb tails and bleed. Melody metrics assume
    one pitch per frame, so overlaps must be resolved before scoring - we keep
    the most confident note and let it win its whole span.
    """
    if not notes:
        return []

    kept: List[Note] = []
    for note in sorted(notes, key=lambda n: -n.confidence):
        if all(note.offset <= k.onset or note.onset >= k.offset for k in kept):
            kept.append(note)

    return sorted(kept, key=lambda n: n.onset)


REGISTRY: Dict[str, Callable[[], System]] = {
    'basic_pitch': BasicPitchSystem,
    'pyin': PyinSystem,
    'ensemble': EnsembleSystem,
    'oracle': OracleSystem,
}


def get_system(name: str, **kwargs) -> System:
    if name not in REGISTRY:
        raise ValueError(f"Unknown system {name!r}. "
                         f"Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
