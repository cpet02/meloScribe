"""The voters: independent opinions about what note is sounding when.

Each one is wrong in a different way, which is the entire point. Measured on
the synthetic benchmark, basic-pitch nails frame pitch (RPA 0.985) but mangles
segmentation (note F1 0.459); PYIN segments well (0.779) but drops octaves
(0.25 on weak-fundamental timbres). Fusing estimators whose errors are
uncorrelated is what buys accuracy - stacking two variants of the same
algorithm would not.

Every voter is optional. A missing dependency disables that voter with a
warning rather than failing the run, so the pipeline degrades in quality
instead of falling over.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..audio import Audio
from .grid import (HOP, MIDI_MAX, MIDI_MIN, N_PITCHES, PITCH_HZ, PITCHES,
                   VoterOutput, gaussian_bump, normalize_rows,
                   resample_matrix, resample_series)


class Voter:
    """Base class: an estimator that reports salience on the shared grid."""

    name = 'base'
    default_weight = 1.0

    def available(self) -> bool:
        return True

    def observe(self, audio: Audio, n_frames: int) -> VoterOutput:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Harmonic template matching - the "play the note and see if it fits" voter
# --------------------------------------------------------------------------

@dataclass
class Timbre:
    """The spectral shape of the instrument we imagine playing each candidate.

    A saxophone is a good template for the sung voice: both are driven
    resonators with strong, regularly-spaced harmonics and a slightly
    odd-weighted spectrum. `odd_boost` above 1 leans it toward the reed's
    closed-pipe character.
    """
    n_harmonics: int = 10
    rolloff: float = 0.72
    odd_boost: float = 1.15

    def weights(self) -> np.ndarray:
        h = np.arange(1, self.n_harmonics + 1, dtype=float)
        w = self.rolloff ** (h - 1)
        w[::2] *= self.odd_boost  # harmonics 1, 3, 5 ... are index 0, 2, 4
        return w / w.sum()


class HarmonicTemplateVoter(Voter):
    """Scores each candidate pitch by how well a synthetic harmonic series
    built on it explains the observed spectrum.

    This is the disciplined form of "simulate a saxophone playing along and
    listen for clashes". Doing it literally - rendering audio and computing
    Plomp-Levelt roughness - would be slow and, worse, nearly octave-blind:
    a tone an octave off is highly consonant, so roughness cannot tell C4
    from C5. That is the exact error we most need to catch.

    Working in the frequency domain fixes both problems. A wrong-octave
    template leaves half its predicted partials sitting on empty spectrum,
    which shows up immediately as unexplained energy. Cost is a handful of
    array lookups per frame.

    The CQT is the right transform here: its bins are log-spaced, so the
    h-th harmonic of any pitch sits a fixed number of bins away
    (round(bins_per_octave * log2(h))) regardless of the fundamental. The
    whole template becomes one strided gather.
    """

    name = 'harmonic_template'
    default_weight = 1.0

    # 3 bins per semitone: enough to see that a note is 30 cents flat without
    # the cost of a full high-resolution transform.
    BINS_PER_OCTAVE = 36
    FMIN_MIDI = 24  # C1, an octave below our lowest candidate, to hold subharmonics

    def __init__(self, timbre: Optional[Timbre] = None,
                 subharmonic_penalty: float = 0.6):
        self.timbre = timbre or Timbre()
        self.subharmonic_penalty = subharmonic_penalty

    def _cqt(self, audio: Audio):
        import librosa

        fmin = float(librosa.midi_to_hz(self.FMIN_MIDI))
        # Cover our highest candidate plus its upper harmonics, stopping below
        # Nyquist so librosa does not complain about unreachable bins.
        max_hz = min(audio.sr / 2 * 0.95, fmin * 2 ** 7.5)
        n_bins = int(np.floor(self.BINS_PER_OCTAVE * np.log2(max_hz / fmin)))
        hop_length = max(1, int(round(HOP * audio.sr)))

        cqt = np.abs(librosa.cqt(
            audio.samples, sr=audio.sr, fmin=fmin, n_bins=n_bins,
            bins_per_octave=self.BINS_PER_OCTAVE, hop_length=hop_length,
        ))
        times = librosa.times_like(cqt, sr=audio.sr, hop_length=hop_length)
        return cqt.T.astype(np.float32), times, n_bins  # (n_frames, n_bins)

    def observe(self, audio: Audio, n_frames: int) -> VoterOutput:
        spec, times, n_bins = self._cqt(audio)
        weights = self.timbre.weights()

        # Bin offset of each harmonic above the fundamental. Constant across
        # pitch because the CQT is log-spaced - this is why the CQT is used.
        harmonic_offsets = np.round(
            self.BINS_PER_OCTAVE * np.log2(np.arange(1, len(weights) + 1))
        ).astype(int)

        # Row index of each candidate pitch within the CQT.
        base_bins = np.round(
            (PITCHES - self.FMIN_MIDI) * (self.BINS_PER_OCTAVE / 12.0)
        ).astype(int)

        # Compress magnitude before summing: without it a single loud partial
        # can carry a template that matches nothing else.
        spec = np.log1p(spec * 20.0)

        salience = np.zeros((spec.shape[0], N_PITCHES), dtype=np.float32)
        for weight, offset in zip(weights, harmonic_offsets):
            bins = base_bins + offset
            valid = bins < n_bins
            salience[:, valid] += weight * spec[:, bins[valid]]

        # Octave defence: if a candidate's own subharmonic (f/2) is strongly
        # present, the energy we are crediting to this pitch is more likely the
        # second harmonic of the note an octave below. Real fundamentals do not
        # sit above a strong f/2; misidentified ones do.
        sub_bins = base_bins - self.BINS_PER_OCTAVE
        sub_valid = sub_bins >= 0
        subharmonic = np.zeros_like(salience)
        subharmonic[:, sub_valid] = spec[:, sub_bins[sub_valid]]
        salience -= self.subharmonic_penalty * subharmonic
        np.maximum(salience, 0.0, out=salience)

        # Total spectral energy is the voicing cue: no energy, nothing sung.
        energy = spec.sum(axis=1)
        voicing = energy / (np.percentile(energy, 95) + 1e-9)
        np.clip(voicing, 0.0, 1.0, out=voicing)

        return VoterOutput(
            name=self.name,
            salience=resample_matrix(normalize_rows(salience), times, n_frames),
            voicing=resample_series(voicing, times, n_frames),
            weight=self.default_weight,
            meta={'timbre': self.timbre.__dict__},
        )


class RoughnessVoter(Voter):
    """The literal reading of the saxophone idea, kept as a diagnostic.

    Sums Plomp-Levelt sensory dissonance between a synthetic reed spectrum on
    each candidate pitch and the observed partials, then inverts it so
    consonance scores high. It is included because it was worth testing rather
    than assuming - but it is off by default and weighted low, because
    consonance is octave-ambiguous by construction (an octave is maximally
    consonant), so it cannot break the tie that matters most.

    Where it does earn its place is confirming a pitch the other voters
    already narrowed down: it responds to genuine spectral clash, so a note
    that fights the accompaniment scores lower even when its partials line up.
    """

    name = 'roughness'
    default_weight = 0.3

    def __init__(self, n_partials: int = 6):
        self.n_partials = n_partials

    @staticmethod
    def _dissonance(f1: np.ndarray, f2: np.ndarray,
                    a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
        """Plomp-Levelt roughness between two sets of partials.

        Peak dissonance near a quarter of a critical bandwidth apart, decaying
        to nothing at unison and at wide separations.
        """
        s = 0.24 / (0.0207 * np.minimum(f1, f2) + 18.96)
        delta = np.abs(f2 - f1)
        return np.minimum(a1, a2) * (
            np.exp(-3.5 * s * delta) - np.exp(-5.75 * s * delta))

    def observe(self, audio: Audio, n_frames: int) -> VoterOutput:
        import librosa

        hop_length = max(1, int(round(HOP * audio.sr)))
        spec = np.abs(librosa.stft(audio.samples, n_fft=2048,
                                   hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=audio.sr, n_fft=2048)
        times = librosa.times_like(spec, sr=audio.sr, hop_length=hop_length)
        spec = spec.T  # (n_frames, n_freq)

        # Only the strongest observed partials matter; the rest is noise floor
        # and would swamp the sum.
        keep = np.argsort(spec, axis=1)[:, -12:]
        rows = np.arange(spec.shape[0])[:, None]
        obs_freqs = freqs[keep]
        obs_amps = spec[rows, keep]
        obs_amps = obs_amps / (obs_amps.max(axis=1, keepdims=True) + 1e-9)

        partials = np.arange(1, self.n_partials + 1, dtype=float)
        partial_amps = 0.7 ** (partials - 1)

        consonance = np.zeros((spec.shape[0], N_PITCHES), dtype=np.float32)
        for p, hz in enumerate(PITCH_HZ):
            probe_freqs = hz * partials
            if probe_freqs[0] > audio.sr / 2:
                continue
            rough = self._dissonance(
                probe_freqs[None, :, None], obs_freqs[:, None, :],
                partial_amps[None, :, None], obs_amps[:, None, :],
            ).sum(axis=(1, 2))
            consonance[:, p] = -rough

        consonance -= consonance.min(axis=1, keepdims=True)
        energy = spec.sum(axis=1)
        voicing = np.clip(energy / (np.percentile(energy, 95) + 1e-9), 0, 1)

        return VoterOutput(
            name=self.name,
            salience=resample_matrix(normalize_rows(consonance), times, n_frames),
            voicing=resample_series(voicing, times, n_frames),
            weight=self.default_weight,
        )


# --------------------------------------------------------------------------
# Established pitch trackers
# --------------------------------------------------------------------------

class PyinVoter(Voter):
    """librosa PYIN. Strong segmentation and voicing, prone to octave slips."""

    name = 'pyin'
    default_weight = 1.0

    def observe(self, audio: Audio, n_frames: int) -> VoterOutput:
        import librosa

        hop_length = max(1, int(round(HOP * audio.sr)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f0, voiced_flag, voiced_prob = librosa.pyin(
                audio.samples,
                fmin=float(librosa.midi_to_hz(MIDI_MIN)),
                fmax=float(librosa.midi_to_hz(MIDI_MAX)),
                sr=audio.sr, hop_length=hop_length,
            )

        times = librosa.times_like(f0, sr=audio.sr, hop_length=hop_length)
        midi = np.full(f0.shape, np.nan)
        valid = ~np.isnan(f0)
        midi[valid] = librosa.hz_to_midi(f0[valid])
        midi = np.nan_to_num(midi, nan=-1.0)

        confidence = np.where(voiced_flag, voiced_prob, 0.0)
        salience = gaussian_bump(midi, confidence)

        return VoterOutput(
            name=self.name,
            salience=resample_matrix(salience, times, n_frames),
            voicing=resample_series(voiced_prob, times, n_frames),
            weight=self.default_weight,
        )


class CrepeVoter(Voter):
    """CREPE via torchcrepe: a CNN trained directly on f0 estimation.

    The strongest single voter available, and the only one whose confidence
    output is genuinely calibrated. Its full 360-bin activation is used rather
    than the argmax frequency, so the decoder sees CREPE's own uncertainty
    instead of a point estimate laundered through a Gaussian.

    Optional: without torchcrepe installed the ensemble runs without it, at a
    measurable accuracy cost.
    """

    name = 'crepe'
    default_weight = 1.6

    # torchcrepe's bin layout is 360 bins of 20 cents, but the base frequency
    # is 10 * 2^(1997.3794/1200) ~ 31.70Hz, not a round musical value. It is
    # read from the library's own converter rather than hard-coded: getting it
    # wrong by the ~54 cents between that and C1 puts every estimate just
    # outside the 50-cent scoring tolerance, which measured as RPA 0.095 -
    # a mapping bug that looks exactly like a broken model.

    def __init__(self, model: str = 'full', batch_size: int = 512,
                 device: Optional[str] = None):
        self.model = model
        self.batch_size = batch_size
        self.device = device

    def available(self) -> bool:
        try:
            import torchcrepe  # noqa: F401
            return True
        except ImportError:
            return False

    def observe(self, audio: Audio, n_frames: int) -> VoterOutput:
        import torch
        import torchcrepe

        device = self.device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # torchcrepe requires exactly 16kHz.
        resampled = audio.resampled(torchcrepe.SAMPLE_RATE)
        tensor = torch.tensor(resampled.samples, dtype=torch.float32)[None]
        hop_length = int(round(HOP * torchcrepe.SAMPLE_RATE))

        # torchcrepe.preprocess is a generator of frame batches, so inference
        # runs batch by batch and the activations are concatenated. Batching is
        # what keeps a long track from trying to hold every frame on the GPU.
        batches = []
        with torch.no_grad():
            for frames in torchcrepe.preprocess(
                    tensor, torchcrepe.SAMPLE_RATE, hop_length,
                    batch_size=self.batch_size, device=device):
                batches.append(
                    torchcrepe.infer(frames, model=self.model,
                                     device=device).cpu())

        if not batches:
            raise RuntimeError('CREPE produced no frames')
        activation = torch.cat(batches).numpy()

        # Map CREPE's bins onto our semitone grid, taking the strongest bin
        # within each semitone rather than the mean: averaging across the 5
        # bins of a semitone would blur a confident peak into its neighbours.
        bin_hz = torchcrepe.convert.bins_to_frequency(
            torch.arange(activation.shape[1])).numpy()
        bin_midi = 69.0 + 12.0 * np.log2(np.maximum(bin_hz, 1e-6) / 440.0)

        salience = np.zeros((activation.shape[0], N_PITCHES), dtype=np.float32)
        for p, midi in enumerate(PITCHES):
            mask = np.abs(bin_midi - midi) <= 0.5
            if np.any(mask):
                salience[:, p] = activation[:, mask].max(axis=1)

        # CREPE's periodicity: the max activation, which is its confidence
        # that a periodic signal is present at all.
        voicing = activation.max(axis=1)
        times = np.arange(activation.shape[0]) * HOP

        return VoterOutput(
            name=self.name,
            salience=resample_matrix(normalize_rows(salience), times, n_frames),
            voicing=resample_series(voicing, times, n_frames),
            weight=self.default_weight,
            meta={'model': self.model, 'device': device},
        )


class BasicPitchVoter(Voter):
    """basic-pitch, using the posteriorgram the old pipeline threw away.

    The previous code kept only the note events and their `amplitude` field,
    calling it confidence - but amplitude is loudness, not probability, which
    is why thresholding on it behaved so erratically. The underlying model
    emits a per-frame, per-pitch activation matrix and a separate onset
    matrix; those are the actual probabilistic outputs, and they are what this
    voter reads. The onset matrix is passed through to the decoder, where it
    informs note boundaries.
    """

    name = 'basic_pitch'
    default_weight = 1.2

    # The model's own layout: 88 piano keys from A0, at ~172 frames/sec.
    MIDI_OFFSET = 21
    FRAME_RATE = 22050 / 256.0

    def available(self) -> bool:
        try:
            import basic_pitch  # noqa: F401
            return True
        except ImportError:
            return False

    def observe(self, audio: Audio, n_frames: int) -> VoterOutput:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict

        if audio.path is None:
            raise ValueError('BasicPitchVoter needs an audio file on disk')

        model_output, _, _ = predict(
            audio_path=str(audio.path),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )

        note_activation = np.asarray(model_output['note'])   # (frames, 88)
        onset_activation = np.asarray(model_output['onset'])

        salience = np.zeros((note_activation.shape[0], N_PITCHES),
                            dtype=np.float32)
        onsets = np.zeros_like(salience)
        for p, midi in enumerate(PITCHES):
            col = int(midi) - self.MIDI_OFFSET
            if 0 <= col < note_activation.shape[1]:
                salience[:, p] = note_activation[:, col]
                onsets[:, p] = onset_activation[:, col]

        times = np.arange(note_activation.shape[0]) / self.FRAME_RATE
        voicing = salience.max(axis=1)

        return VoterOutput(
            name=self.name,
            salience=resample_matrix(salience, times, n_frames),
            voicing=resample_series(voicing, times, n_frames),
            weight=self.default_weight,
            meta={'onsets': resample_matrix(onsets, times, n_frames)},
        )


# The best voter set depends on whether CREPE is installed, and the two
# answers are not nested - so the default is resolved at runtime rather than
# fixed. Measured on the synthetic benchmark (10 tracks, OA / note F1 / secs):
#
#   crepe + basic_pitch                       0.973 / 0.919 /  5.4   <- with
#   crepe + basic_pitch + pyin + harmonic     0.965 / 0.919 / 35.4
#   crepe alone                               0.968 / 0.945 /  8.0
#   basic_pitch + pyin + harmonic_template    0.946 / 0.919 / 29.9   <- without
#   basic_pitch alone                         0.933 / 0.909 /  1.5
#
# Two findings worth recording. PYIN actively *hurts* once CREPE is present
# and accounts for ~29s of that 35s, so it is dropped from the CREPE path and
# kept only as a fallback. And CREPE alone has the best note F1 overall but
# drops to 0.727 on re-articulation cases, because nothing tells it that three
# repeated notes are not one long one - basic-pitch's onset matrix is what
# fixes that, which is why it is in both sets.
AUTO = 'auto'
DEFAULT_VOTERS = AUTO

VOTERS_WITH_CREPE = ('crepe', 'basic_pitch')
VOTERS_WITHOUT_CREPE = ('basic_pitch', 'pyin', 'harmonic_template')

_REGISTRY = {
    'crepe': CrepeVoter,
    'basic_pitch': BasicPitchVoter,
    'pyin': PyinVoter,
    'harmonic_template': HarmonicTemplateVoter,
    'roughness': RoughnessVoter,
}


def resolve_voter_names(names=AUTO) -> tuple:
    """Turn a request (possibly 'auto') into a concrete voter list."""
    if names is None or names == AUTO:
        return (VOTERS_WITH_CREPE if CrepeVoter().available()
                else VOTERS_WITHOUT_CREPE)
    return tuple(names)


def build_voters(names=AUTO, warn_missing: bool = True) -> List[Voter]:
    """Instantiate the named voters, silently dropping unavailable ones."""
    voters: List[Voter] = []
    for name in resolve_voter_names(names):
        if name not in _REGISTRY:
            raise ValueError(f"Unknown voter {name!r}. "
                             f"Available: {sorted(_REGISTRY)}")
        voter = _REGISTRY[name]()
        if not voter.available():
            if warn_missing:
                warnings.warn(
                    f"Voter {name!r} unavailable (missing dependency) - "
                    f"continuing without it, which will cost accuracy.")
            continue
        voters.append(voter)

    if not voters:
        raise RuntimeError('No pitch voters available.')
    return voters
