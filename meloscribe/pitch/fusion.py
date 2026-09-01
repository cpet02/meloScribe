"""Fusing the voters into one decoded melody line.

Two ideas do the work here.

First, **combine before deciding**. The old pipeline thresholded a single
estimator's output frame by frame, so one bad frame became a spurious note.
Here every voter contributes log-evidence for every candidate pitch, and
nothing is committed to until all of them have spoken.

Second, **decode with a transition model**. A melody is not a sequence of
independent frames: it holds notes, and it moves by small intervals far more
often than large ones. A Viterbi pass over pitch states encodes exactly that,
which is what lets strong evidence either side of a corrupted frame carry the
note through it intact.

Confidence then comes from forward-backward, not from the Viterbi score: the
posterior probability of the decoded state given the *whole* signal. That is a
genuinely calibrated number - unlike the note amplitude the old code reported
as confidence - so a threshold on it means something consistent across songs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .grid import HOP, N_PITCHES, PITCHES, VoterOutput

# The unvoiced state lives at the end of the state vector, so state index
# N_PITCHES means "nobody is singing".
UNVOICED = N_PITCHES
N_STATES = N_PITCHES + 1

_LOG_EPS = 1e-8


@dataclass
class FusionSettings:
    """Knobs for the decoder. Defaults were tuned on the synthetic benchmark."""

    # Cost per semitone of movement between frames. Higher means a smoother,
    # more sustained melody; too high and real leaps get flattened.
    step_penalty: float = 0.55
    # Beyond this interval, extra distance stops mattering - an octave leap and
    # a tenth leap are both simply "a leap", and taxing them by raw distance
    # would make wide-range singing systematically unlikely.
    max_free_leap: float = 12.0
    leap_cap: float = 7.0
    # Log-odds bonus for staying on the current pitch. This is what turns
    # frame estimates into sustained notes.
    sustain_bonus: float = 2.2
    # Cost of starting or stopping singing. Discourages the one-frame
    # dropouts that fragment a held note into three.
    voicing_switch_penalty: float = 2.0
    # Raises or lowers the bar for calling a frame voiced at all. Positive
    # values make the decoder more willing to say "silence".
    silence_bias: float = 0.0
    # Floor applied to every voter's salience before the log. Without it a
    # voter that reports ~0 for a pitch contributes log(1e-8) ~ -18, which no
    # amount of contrary evidence can outweigh - one voter's mistake silently
    # becomes a veto. Measured: the unfloored fusion scored *below* its own
    # best single voter (OA 0.848 vs 0.920), which is the signature of exactly
    # that failure. The floor bounds any single voter's dissent, so fusion
    # aggregates opinions instead of letting the most confident one decide.
    voter_floor: float = 0.08
    # Weight of the musical-key prior, in log-odds. Deliberately small: this
    # nudges, it does not filter. The previous pipeline deleted out-of-key
    # notes outright, which destroyed every accidental in the song.
    key_prior_weight: float = 0.35


@dataclass
class DecodedFrame:
    """Per-frame decoder output, retained for the UI's confidence display."""
    times: np.ndarray
    midi: np.ndarray          # -1 where unvoiced
    confidence: np.ndarray    # posterior probability of the decoded state
    voiced: np.ndarray
    per_voter: Dict[str, np.ndarray] = field(default_factory=dict)


def fuse_observations(outputs: Sequence[VoterOutput],
                      settings: Optional[FusionSettings] = None,
                      key_prior: Optional[np.ndarray] = None) -> np.ndarray:
    """Combine voters into a per-frame log-likelihood over states.

    Combination is a weighted geometric mean - a sum in log space. That makes
    a voter's confident *rejection* of a pitch count as much as another's
    endorsement, which is precisely how the harmonic-template voter suppresses
    PYIN's octave errors: PYIN votes for the wrong octave, the template finds
    half its predicted partials missing there, and the product collapses.
    """
    settings = settings or FusionSettings()
    if not outputs:
        raise ValueError('No voter outputs to fuse')

    n_frames = outputs[0].n_frames
    total_weight = sum(o.weight for o in outputs)

    log_pitch = np.zeros((n_frames, N_PITCHES), dtype=np.float64)
    voicing = np.zeros(n_frames, dtype=np.float64)

    for out in outputs:
        if out.n_frames != n_frames:
            raise ValueError(
                f"Voter {out.name} has {out.n_frames} frames, expected {n_frames}")
        share = out.weight / total_weight
        floored = np.maximum(out.salience.astype(np.float64),
                             settings.voter_floor)
        log_pitch += share * np.log(floored)
        voicing += share * out.voicing.astype(np.float64)

    if key_prior is not None:
        log_pitch += settings.key_prior_weight * key_prior[None, :]

    # The unvoiced state competes directly against the best pitch: its score is
    # the log-odds of silence, placed on the same scale as the pitch evidence.
    voicing = np.clip(voicing, _LOG_EPS, 1.0 - _LOG_EPS)
    log_unvoiced = np.log(1.0 - voicing) - settings.silence_bias
    log_voiced_offset = np.log(voicing)

    observations = np.empty((n_frames, N_STATES), dtype=np.float64)
    observations[:, :N_PITCHES] = log_pitch + log_voiced_offset[:, None]
    observations[:, UNVOICED] = log_unvoiced
    return observations


def build_transition_matrix(settings: Optional[FusionSettings] = None) -> np.ndarray:
    """Log-probability of moving between any two states in one 10ms frame."""
    settings = settings or FusionSettings()
    transition = np.zeros((N_STATES, N_STATES), dtype=np.float64)

    interval = np.abs(PITCHES[:, None] - PITCHES[None, :])
    # Cost grows with interval size but saturates, so wide leaps stay possible.
    cost = settings.step_penalty * np.minimum(interval, settings.max_free_leap)
    cost = np.minimum(cost, settings.leap_cap)
    np.fill_diagonal(cost, -settings.sustain_bonus)

    transition[:N_PITCHES, :N_PITCHES] = -cost
    transition[:N_PITCHES, UNVOICED] = -settings.voicing_switch_penalty
    transition[UNVOICED, :N_PITCHES] = -settings.voicing_switch_penalty
    transition[UNVOICED, UNVOICED] = settings.sustain_bonus

    # Normalise each row to a proper distribution, so the forward-backward
    # posteriors are probabilities rather than arbitrary scores.
    transition -= _logsumexp(transition, axis=1, keepdims=True)
    return transition


def _logsumexp(x: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    peak = np.max(x, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    out = peak + np.log(np.sum(np.exp(x - peak), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def viterbi(observations: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Most likely state sequence given the observations."""
    n_frames = observations.shape[0]
    if n_frames == 0:
        return np.zeros(0, dtype=int)

    scores = observations[0].copy()
    backpointers = np.zeros((n_frames, N_STATES), dtype=np.int32)

    for t in range(1, n_frames):
        candidates = scores[:, None] + transition
        backpointers[t] = np.argmax(candidates, axis=0)
        scores = candidates[backpointers[t], np.arange(N_STATES)] + observations[t]

    path = np.zeros(n_frames, dtype=int)
    path[-1] = int(np.argmax(scores))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = backpointers[t, path[t]]
    return path


def forward_backward(observations: np.ndarray,
                     transition: np.ndarray) -> np.ndarray:
    """Posterior probability of each state at each frame.

    This is where calibrated confidence comes from. The Viterbi path says what
    the melody most likely is; the posterior says how sure we are, accounting
    for evidence both before and after the frame in question. A note that is
    only marginally preferred over its neighbour reports low confidence even
    though it won the path - exactly the case a user should see flagged.
    """
    n_frames = observations.shape[0]
    if n_frames == 0:
        return np.zeros((0, N_STATES))

    log_alpha = np.zeros((n_frames, N_STATES))
    log_alpha[0] = observations[0] - _logsumexp(observations[0])
    for t in range(1, n_frames):
        log_alpha[t] = observations[t] + _logsumexp(
            log_alpha[t - 1][:, None] + transition, axis=0)
        log_alpha[t] -= _logsumexp(log_alpha[t])

    log_beta = np.zeros((n_frames, N_STATES))
    for t in range(n_frames - 2, -1, -1):
        log_beta[t] = _logsumexp(
            transition + (observations[t + 1] + log_beta[t + 1])[None, :], axis=1)
        log_beta[t] -= _logsumexp(log_beta[t])

    log_posterior = log_alpha + log_beta
    log_posterior -= _logsumexp(log_posterior, axis=1, keepdims=True)
    return np.exp(log_posterior)


def decode(outputs: Sequence[VoterOutput],
           settings: Optional[FusionSettings] = None,
           key_prior: Optional[np.ndarray] = None) -> DecodedFrame:
    """Run the full fuse -> decode -> score chain."""
    settings = settings or FusionSettings()
    observations = fuse_observations(outputs, settings, key_prior)
    transition = build_transition_matrix(settings)

    path = viterbi(observations, transition)
    posterior = forward_backward(observations, transition)

    n_frames = observations.shape[0]
    times = np.arange(n_frames) * HOP
    voiced = path != UNVOICED
    midi = np.where(voiced, PITCHES[np.clip(path, 0, N_PITCHES - 1)], -1.0)
    confidence = posterior[np.arange(n_frames), path]

    per_voter = {
        out.name: out.salience[np.arange(n_frames), np.clip(path, 0, N_PITCHES - 1)]
        for out in outputs
    }

    return DecodedFrame(times=times, midi=midi, confidence=confidence,
                        voiced=voiced, per_voter=per_voter)


def key_prior_vector(pitch_classes: Sequence[int],
                     in_key: float = 1.0,
                     out_of_key: float = -1.0) -> np.ndarray:
    """A gentle per-pitch log-odds bias from an estimated key.

    Returned as a bias, never a mask. An out-of-key note with strong acoustic
    evidence still wins - accidentals, blue notes and chromatic passing tones
    are real, and a hard key filter erases every one of them.
    """
    allowed = set(int(p) % 12 for p in pitch_classes)
    return np.array([in_key if int(p) % 12 in allowed else out_of_key
                     for p in PITCHES], dtype=np.float64)
