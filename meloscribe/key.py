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

# How each pitch class is named under the two kinds of key signature. Where
# there is no signature to follow - C major, A minor, or no key worth
# trusting - notes are named with sharps, the usual default for pitch names.
SPELLINGS = {
    'sharp': ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'),
    'flat': ('C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'),
}

# Each letter's natural pitch class, in scale order: for the one note a key
# spells from a given letter rather than from either table above.
NATURALS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

# The major keys whose signature is flats: F, and every tonic on a black key.
# Two of those have an enharmonic twin. Db (five flats) beats C# (seven
# sharps); Gb against F# is a genuine six-six tie, settled as Gb so the rule
# stays "F and the black keys" - which also names its relative minor Eb
# minor, the commoner spelling, rather than D# minor.
FLAT_MAJOR_TONICS = frozenset({1, 3, 5, 6, 8, 10})

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
        # Spelled by its own signature: Eb major, never D# major.
        tonic = SPELLINGS[self.spelling][self.tonic]
        return f"{tonic} {'major' if self.is_major else 'minor'}"

    @property
    def spelling(self) -> str:
        """'flat' or 'sharp': which accidentals this key's signature uses."""
        # A minor key shares its signature with the major a minor third up.
        relative_major = self.tonic if self.is_major else (self.tonic + 3) % 12
        return 'flat' if relative_major in FLAT_MAJOR_TONICS else 'sharp'

    @property
    def pitch_names(self) -> Tuple[str, ...]:
        """What this key calls each pitch class, index = pitch class.

        Its signature's accidentals throughout, except a minor key's leading
        tone, which is always written as the raised 7th. Minor melodies lean
        on it, and D minor's flats alone would call it Db - which reads as a
        wrong note, where C# reads as the leading tone it is.
        """
        names = list(SPELLINGS[self.spelling])
        if not self.is_major:
            names[(self.tonic - 1) % 12] = self._raised_seventh()
        return tuple(names)

    def _raised_seventh(self) -> str:
        # The letter below the tonic's, sharpened as far as a semitone below
        # the tonic needs: B in C minor, C# in D minor, F## in G# minor.
        tonic_letter = SPELLINGS[self.spelling][self.tonic][0]
        letters = list(NATURALS)
        letter = letters[letters.index(tonic_letter) - 1]
        return letter + '#' * ((self.tonic - 1 - NATURALS[letter]) % 12)

    def transposed(self, semitones: int) -> 'KeyEstimate':
        """The key as a transposing instrument reads it: concert Gb major is
        Eb major for alto sax (+9)."""
        return KeyEstimate(tonic=(self.tonic + semitones) % 12,
                           is_major=self.is_major, confidence=self.confidence)

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


def note_spelling(key: Optional[KeyEstimate], transpose: int = 0) -> str:
    """'flat' or 'sharp': the accidentals of the key that the notes of a song
    in `key` are written in, once transposed by `transpose` semitones.

    It is the written key that decides, and transposition can flip it:
    concert Bb major, a flat key, is G major on alto sax and takes sharps.
    """
    written = _spelling_key(key, transpose)
    return written.spelling if written else 'sharp'


def note_names(key: Optional[KeyEstimate],
               transpose: int = 0) -> Tuple[str, ...]:
    """What to call each pitch class of those same notes, index = pitch
    class: the written key's `pitch_names`, on the same terms as
    `note_spelling`, so the table and the flag cannot disagree."""
    written = _spelling_key(key, transpose)
    return written.pitch_names if written else SPELLINGS['sharp']


def _spelling_key(key: Optional[KeyEstimate],
                  transpose: int) -> Optional[KeyEstimate]:
    # A key too uncertain to use as the decoder's prior is too uncertain to
    # choose accidentals by: None, and the notes get the same sharps as no key.
    if key is None or key.as_prior() is None:
        return None
    return key.transposed(transpose)


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
