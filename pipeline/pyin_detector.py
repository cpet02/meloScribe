"""
Phase 2c (supplement): PYIN Pitch Detection
Runs librosa's PYIN algorithm on a vocal stem as a second pitch estimator.
Used to cross-validate note events produced by the basic-pitch pipeline.
"""

import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


class PYINDetector:
    """Detects pitch using the PYIN algorithm via librosa."""

    def detect_pitch_pyin(self, audio_path: str) -> List[Dict[str, Any]]:
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"  Detecting PYIN pitch from: {audio_path.name}")

        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        frames = self._run_pyin(y, sr)

        print(f"  PYIN: {len(frames)} voiced frames detected")
        return frames

    def _run_pyin(self, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        Run librosa.pyin() and return voiced frames only.

        Downsamples to 16kHz and uses a larger hop_length for speed —
        PYIN is only used for note-level cross-validation so full
        resolution is unnecessary and just wastes time.
        """
        # Downsample to 16kHz — halves processing time with no meaningful
        # accuracy loss for cross-validation purposes
        target_sr = 16000
        if sr > target_sr:
            y  = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # 1024 at 16kHz ≈ 64ms per frame — coarse but plenty for note matching
        hop_length = 1024

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=hop_length,
        )

        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

        frames = []
        for t, freq, voiced, prob in zip(times, f0, voiced_flag, voiced_probs):
            if voiced and freq is not None and not np.isnan(freq):
                frames.append({
                    'time':       float(t),
                    'frequency':  float(freq),
                    'confidence': float(prob),
                })

        return frames


def get_pyin_pitch(audio_path: str) -> List[Dict[str, Any]]:
    return PYINDetector().detect_pitch_pyin(audio_path)
