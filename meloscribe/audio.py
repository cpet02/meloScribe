"""Audio loading and conditioning shared by every stage.

Centralised so that "what sample rate is this array at?" has exactly one answer
per stage, rather than each module quietly resampling to its own preference and
misaligning the timestamps downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Pitch analysis runs at 16kHz: every voter wants it, it is CREPE's native
# rate, and nothing useful for f0 estimation lives above 8kHz.
ANALYSIS_SR = 16000


@dataclass
class Audio:
    """A mono waveform with its sample rate attached."""
    samples: np.ndarray
    sr: int
    path: Optional[Path] = None

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr

    def resampled(self, target_sr: int) -> 'Audio':
        if target_sr == self.sr:
            return self
        import librosa
        return Audio(
            samples=librosa.resample(self.samples, orig_sr=self.sr,
                                     target_sr=target_sr),
            sr=target_sr,
            path=self.path,
        )


def load(path, sr: Optional[int] = None, mono: bool = True) -> Audio:
    """Load any audio file librosa can read, optionally resampling."""
    import librosa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    samples, actual_sr = librosa.load(str(path), sr=sr, mono=mono)
    return Audio(samples=np.asarray(samples, dtype=np.float32),
                 sr=int(actual_sr), path=path)


def normalize(audio: Audio, target_peak: float = 0.95) -> Audio:
    """Peak-normalise, leaving digital silence alone."""
    peak = float(np.max(np.abs(audio.samples))) if audio.samples.size else 0.0
    if peak <= 1e-6:
        return audio
    return Audio(samples=audio.samples * (target_peak / peak),
                 sr=audio.sr, path=audio.path)


def bandpass_voice(audio: Audio, low_hz: float = 60.0,
                   high_hz: float = 8000.0) -> Audio:
    """Restrict to the range a sung melody can actually occupy.

    Sub-bass below 60Hz is bass-stem bleed or rumble, never a sung fundamental;
    removing it stops pitch trackers from being pulled an octave down. The
    upper bound only trims hiss, since the fundamental is far below it.
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = audio.sr / 2
    high_hz = min(high_hz, nyquist * 0.99)
    if low_hz >= high_hz:
        return audio

    sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype='band',
                 output='sos')
    # Zero-phase, so filtering introduces no time offset - critical when the
    # whole point of the pipeline is timestamps.
    filtered = sosfiltfilt(sos, audio.samples).astype(np.float32)
    return Audio(samples=filtered, sr=audio.sr, path=audio.path)


def suppress_percussive(audio: Audio, margin: float = 2.0) -> Audio:
    """Keep the harmonic component, discarding percussive bleed.

    Even a good vocal stem carries snare and hi-hat leakage, which reads as a
    broadband transient and triggers spurious onsets. HPSS removes most of it
    while leaving the sung tone intact.
    """
    import librosa
    harmonic = librosa.effects.harmonic(audio.samples, margin=margin)
    return Audio(samples=np.asarray(harmonic, dtype=np.float32),
                 sr=audio.sr, path=audio.path)


def prepare_for_pitch(path, denoise: bool = True) -> Audio:
    """The standard conditioning chain applied before pitch analysis."""
    audio = load(path, sr=ANALYSIS_SR, mono=True)
    audio = bandpass_voice(audio)
    if denoise:
        audio = suppress_percussive(audio)
    return normalize(audio)


def rms_envelope(audio: Audio, hop: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Frame-level loudness, used as a voicing prior and for silence gating."""
    import librosa

    hop_length = max(1, int(round(hop * audio.sr)))
    rms = librosa.feature.rms(y=audio.samples, hop_length=hop_length,
                              frame_length=hop_length * 4)[0]
    times = librosa.times_like(rms, sr=audio.sr, hop_length=hop_length)
    return times, rms


def write(path, audio: Audio) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.samples, audio.sr)
    return path
