"""Key estimation, used as a prior rather than a filter.

Krumhansl-Schmuckler: correlate the track's chroma profile against empirically
derived major and minor key profiles, and take the best match.

The important design point is what happens with the answer. The old pipeline
used the key to *delete* notes outside the scale, which throws away every
accidental, blue note and chromatic passing tone in the song - and does so
irreversibly, on the strength of an estimate that is itself often wrong. Here
the key only biases the decoder's log-probabilities, so out-of-key notes
survive when the acoustic evidence supports them. The estimate is also
reported with its own confidence, and a weak estimate contributes
proportionally less.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

PITCH_CLASSES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

# Krumhansl-Kessler probe-tone profiles: how well each scale degree fits a key,
# derived from listener ratings rather than theory.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
# Natural minor plus the raised 6th and 7th, since both appear constantly in
# real melodies and treating them as out-of-key would penalise correct notes.
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 9, 10, 11)


@dataclass
class KeyEstimate:
    tonic: int
    is_major: bool
    confidence: float

    @property
    def name(self) -> str:
        return f"{PITCH_CLASSES[self.tonic]} {'major' if self.is_major else 'minor'}"

    @property
    def pitch_classes(self) -> List[int]:
        scale = MAJOR_SCALE if self.is_major else MINOR_SCALE
        return [(self.tonic + step) % 12 for step in scale]

    def as_prior(self, min_confidence: float = 0.55) -> Optional[List[int]]:
        """The scale to bias toward, or None when the estimate is too weak.

        Below the threshold the estimate carries no useful information, and
        biasing on a coin-flip would inject noise into the decoder.
        """
        return self.pitch_classes if self.confidence >= min_confidence else None


def chroma_profile(audio_path, sr: int = 22050) -> np.ndarray:
    """Duration-weighted chroma for a whole track."""
    import librosa

    y, actual_sr = librosa.load(str(audio_path), sr=sr, mono=True)
    # CQT chroma rather than STFT: log-spaced bins align with semitones, so
    # pitch classes are not smeared across neighbours in the bass register.
    chroma = librosa.feature.chroma_cqt(y=y, sr=actual_sr)
    profile = chroma.mean(axis=1)
    total = profile.sum()
    return profile / total if total > 0 else profile


def estimate_from_profile(profile: np.ndarray) -> KeyEstimate:
    """Correlate a chroma profile against all 24 keys."""
    scores: List[Tuple[float, int, bool]] = []

    for tonic in range(12):
        for is_major in (True, False):
            reference = np.roll(MAJOR_PROFILE if is_major else MINOR_PROFILE, tonic)
            correlation = float(np.corrcoef(profile, reference)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
            scores.append((correlation, tonic, is_major))

    scores.sort(reverse=True)
    best, runner_up = scores[0], scores[1]

    # Confidence is the margin over the second-best key, not the raw
    # correlation. Every key correlates decently with tonal music; what tells
    # us we have the right one is that it beats its closest rival - typically
    # the relative major/minor, which shares all seven notes.
    margin = best[0] - runner_up[0]
    confidence = float(np.clip(best[0] * 0.5 + margin * 2.5, 0.0, 1.0))

    return KeyEstimate(tonic=best[1], is_major=best[2], confidence=confidence)


def estimate_key(audio_path) -> KeyEstimate:
    """Estimate the key of a track."""
    return estimate_from_profile(chroma_profile(audio_path))


def estimate_from_notes(notes) -> Optional[KeyEstimate]:
    """Estimate key from transcribed notes instead of raw audio.

    Weighted by duration and confidence: a long, confidently detected note is
    much stronger evidence of the key than a brief uncertain one.
    """
    if not notes:
        return None

    profile = np.zeros(12)
    for note in notes:
        profile[int(note.midi) % 12] += note.duration * note.confidence

    total = profile.sum()
    if total <= 0:
        return None
    return estimate_from_profile(profile / total)
