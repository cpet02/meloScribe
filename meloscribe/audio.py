"""Audio loading and conditioning shared by every stage.

Centralised so that "what sample rate is this array at?" has exactly one answer
per stage, rather than each module quietly resampling to its own preference and
misaligning the timestamps downstream.
"""

from __future__ import annotations

import atexit
import hashlib
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Pitch analysis runs at 16kHz: every voter wants it, it is CREPE's native
# rate, and nothing useful for f0 estimation lives above 8kHz.
ANALYSIS_SR = 16000


class AudioLoadError(ValueError):
    """The input could not be decoded into usable audio.

    Carries the file name and the decoder's reason, instead of whatever the
    backend threw - for an undecodable MP3 that was audioread's NoBackendError
    with an empty message, and for an empty WAV a scipy padding error.
    """


@dataclass
class Audio:
    """A mono waveform with its sample rate attached."""
    samples: np.ndarray
    sr: int
    path: Optional[Path] = None
    # What loading had to fix to make the file usable (non-finite samples, a
    # downmix that cancelled). Empty for ordinary input.
    repairs: List[str] = field(default_factory=list)

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
    """Load any audio file librosa can read, optionally resampling.

    Two kinds of damage are repaired, with a warning and a note in
    `Audio.repairs`, rather than crashing or silently emptying the run: a
    non-finite sample (one NaN made librosa.resample reject the whole file),
    and channels that cancel when summed (a phase-inverted stereo vocal
    downmixes to silence, which normalisation then amplifies into noise).
    """
    import librosa

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        # Native rate, all channels: the repairs must precede resampling.
        samples, native_sr = librosa.load(str(path), sr=None, mono=False)
    except Exception as exc:
        raise AudioLoadError(f"Could not decode {path.name}: "
                             f"{str(exc) or type(exc).__name__}") from exc
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise AudioLoadError(f"{path.name} contains no audio")

    repairs: List[str] = []
    finite = np.isfinite(samples)
    if not finite.all():
        repairs.append(f"{int((~finite).sum())} non-finite sample(s) set to 0")
        samples = np.where(finite, samples, 0.0).astype(np.float32)

    if mono and samples.ndim > 1:
        mixed = samples.mean(axis=0)
        channel_rms = np.sqrt(np.mean(samples ** 2, axis=1))
        # Only near-total cancellation (>20 dB) counts: uncorrelated or
        # hard-panned channels lose at most 6 dB in a downmix.
        if np.sqrt(np.mean(mixed ** 2)) < 0.1 * channel_rms.max():
            repairs.append('channels cancel when summed (phase-inverted?); '
                           'using the loudest channel')
            mixed = samples[int(np.argmax(channel_rms))]
        samples = mixed

    if sr is not None and sr != native_sr:
        samples = librosa.resample(samples, orig_sr=native_sr, target_sr=sr)
        native_sr = sr

    if repairs:
        warnings.warn(f"{path.name}: " + '; '.join(repairs))
    return Audio(samples=np.asarray(samples, dtype=np.float32),
                 sr=int(native_sr), path=path, repairs=repairs)


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
    # whole point of the pipeline is timestamps. The edge padding is scipy's
    # default, capped for clips of a few milliseconds that are shorter than it.
    padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                         (sos[:, 5] == 0).sum()))
    filtered = sosfiltfilt(sos, audio.samples,
                           padlen=max(0, min(padlen, audio.samples.size - 1))
                           ).astype(np.float32)
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
    repairs = audio.repairs
    audio = bandpass_voice(audio)
    if denoise:
        audio = suppress_percussive(audio)
    audio = normalize(audio)
    audio.repairs = repairs
    return audio


_REPAIRED_DIR: Optional[Path] = None


def usable_path(audio: Audio, original) -> Path:
    """The file a disk-reading voter (basic-pitch) should open for `audio`.

    Normally the original. When loading had to repair it, the original would
    crash such a voter (NaN) or feed it silence (cancelling channels), so it
    gets a copy of the repaired audio instead, deleted when the process exits.
    """
    if not audio.repairs:
        return Path(original)
    global _REPAIRED_DIR
    if _REPAIRED_DIR is None:
        _REPAIRED_DIR = Path(tempfile.mkdtemp(prefix='meloscribe-repaired-'))
        atexit.register(shutil.rmtree, _REPAIRED_DIR, True)
    digest = hashlib.sha1(audio.samples.tobytes()).hexdigest()[:16]
    return write(_REPAIRED_DIR / f"{digest}.wav", audio)


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
