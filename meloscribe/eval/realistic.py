"""A realistic sung-vocal benchmark: a real voice, a band, an MP3, exact truth.

    python -m meloscribe.eval.realistic generate [--data-dir DIR]
    python -m meloscribe.eval.realistic score --systems basic_pitch,ensemble --path clean,proxy
    python -m meloscribe.eval.realistic score --systems oracle        # self-test, must be 1.000

Why this exists
---------------
`synth.py` renders additive harmonic tones and note F1 is 1.000 on all 13 of
its cases, so it can no longer tell good from better - and by decision 7 a
benchmark that cannot see a failure will argue against fixing it. Real singing
differs from those tones in exactly the ways a transcriber finds hard:
consonants interrupt voicing, formants move every syllable, onsets are breathy,
vibrato is wide and arrives late, notes are scooped into, intonation drifts,
reverb rings past the offset, backing vocals sit in the same stem, and the stem
itself is a separator's imperfect guess. This set puts all of that back while
keeping the one property a real recording cannot give: truth known to the frame.

Technique
---------
WORLD analysis/resynthesis (pyworld) of a real human voice - the CMU ARCTIC
utterance that pysptk ships as example data - with the melody's f0 imposed.
This is the MDB-stem-synth technique: spectral envelope (formants, consonants,
breath) and aperiodicity come from a real recording; the f0 is ours, so it is
exact. The utterance is cut into vowel nuclei, voiced consonants and
fricatives, and each sung syllable is re-assembled from those units under its
notes. When no recording can be found, an analytic formant voice (a plain
source-filter model) is used instead, with a warning; the smoke test uses it
deliberately so that it needs no recording.

Ground-truth conventions
------------------------
- Only the lead melody is annotated. Backing vocals, rapped speech and the
  instrumental lead are in the audio and deliberately absent from the truth.
- A note starts where its voicing starts: after an unvoiced consonant, at the
  start of a voiced one, at the start of a scoop. Between legato notes inside a
  syllable the boundary is the centre of the pitch glide.
- A note's pitch is the intended semitone. The f0 contour carries what was
  actually sung: vibrato, scoops, glides, drift, jitter and falls.

Requirements: `pip install pyworld` (generation), soundfile>=0.12 (its wheels
bundle libsndfile>=1.1, which encodes and decodes MP3 - verified sample-aligned),
scipy, librosa, mir_eval. pysptk is optional: only its bundled CMU ARCTIC WAV is
used, so `pip install pysptk` for the real voice (or point MELOSCRIBE_VOICE_WAV
at any clean voice recording). Nothing is downloaded automatically.

Running it: CREPE `full` dominates (~97% of ensemble time) and runs ~6-12x
slower than real time on CPU, so score on a GPU. Voter outputs are cached by
content under <data-dir>/voter_cache, so re-scoring after a fusion or
segmentation change costs seconds per case; editing voters.py, grid.py or
audio.py invalidates the cache automatically.

Evaluation paths (Demucs cannot run everywhere, so the stem is modelled):
  clean   the vocal stem itself (lead + backing + reverb): the upper bound
  proxy   the decoded MP3 mix through a smeared, floored ideal ratio mask - a
          separator with accompaniment residual at -12..-24 dB
  repet   librosa's REPET-SIM vocal separation of the MP3: a pessimistic floor
  demucs  real Demucs on the MP3, when it is installed
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import logging
import math
import multiprocessing
import os
import shutil
import sys
import time
import warnings
import zlib
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .groundtruth import DEFAULT_HOP, GroundTruth, Note, Prediction, resample_f0

SR = 44100
WORLD_SR = 16000
FRAME_S = 0.005
MP3_KBPS = 128
GENERATOR_VERSION = 1
DEFAULT_DATA_DIR = Path(os.environ.get('MELOSCRIBE_REALISTIC_DIR',
                                       'data/eval/realistic'))
PATHS = ('clean', 'proxy', 'repet', 'demucs')
DEFAULT_PATHS = ('clean', 'proxy')
SPURIOUS_MAX_S = 0.15
ONSET_TOL = 0.05
PITCH_TOL_CENTS = 50.0


def _seed(*parts) -> int:
    """Stable across processes - never `hash()` (see synth.case_seed)."""
    return zlib.crc32('/'.join(str(p) for p in parts).encode('utf-8'))


def _hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def _ramp(n: int) -> np.ndarray:
    """Raised-cosine rise from ~0 to exactly 1 over n frames."""
    if n <= 0:
        return np.zeros(0)
    return 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 1) / n)


def _smooth_noise(rng, n: int, width: int) -> np.ndarray:
    """Unit-variance Gaussian noise low-passed by a Hann window."""
    x = rng.standard_normal(n + 2 * width)
    k = np.hanning(2 * width + 1)
    y = np.convolve(x, k / k.sum(), mode='same')[width:width + n]
    sd = float(y.std())
    return y / sd if sd > 0 else y


def _runs(mask: np.ndarray):
    start = 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            yield start, i, bool(mask[start])
            start = i


# --------------------------------------------------------------------------
# The voice: WORLD units cut from a real recording
# --------------------------------------------------------------------------

@dataclass
class VoiceBank:
    sp: np.ndarray                  # (frames, bins) power spectral envelope
    ap: np.ndarray                  # aperiodicity, same shape
    f0: np.ndarray                  # the recording's own f0 (spoken sections)
    nuclei: List[Tuple[int, int]]   # stable vowel frames
    voiced_cons: List[Tuple[int, int]]
    fricatives: List[Tuple[int, int]]
    source: str
    frame_power: np.ndarray = field(default=None)
    ref_power: float = 1.0

    def __post_init__(self) -> None:
        self.frame_power = self.sp.sum(axis=1) + 1e-30
        frames = np.concatenate([np.arange(a, b) for a, b in self.nuclei])
        self.ref_power = float(np.median(self.frame_power[frames]))

    @property
    def n_bins(self) -> int:
        return self.sp.shape[1]

    def warped(self, alpha: float) -> 'VoiceBank':
        """Scale the vocal tract: alpha > 1 moves every formant up (shorter
        tract), which is most of what separates a female voice from a male one
        once the f0 is imposed."""
        if abs(alpha - 1.0) < 1e-6:
            return self
        return replace(self, sp=_warp_bins(self.sp, alpha),
                       ap=_warp_bins(self.ap, alpha))


def _warp_bins(mat: np.ndarray, alpha: float) -> np.ndarray:
    n = mat.shape[1]
    src = np.clip(np.arange(n) / alpha, 0, n - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = src - lo
    return mat[:, lo] * (1 - frac) + mat[:, hi] * frac


def analyse_recording(path) -> VoiceBank:
    import pyworld as pw
    import soundfile as sf
    from scipy.signal import resample_poly

    x, fs = sf.read(str(path), always_2d=True)
    x = x.mean(axis=1)
    if fs != WORLD_SR:
        g = math.gcd(int(fs), WORLD_SR)
        x = resample_poly(x, WORLD_SR // g, int(fs) // g)
    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, WORLD_SR, f0_floor=60.0, f0_ceil=600.0,
                       frame_period=FRAME_S * 1000)
    sp = pw.cheaptrick(x, f0, t, WORLD_SR)
    ap = pw.d4c(x, f0, t, WORLD_SR)
    return _segment_bank(sp, ap, f0, Path(path).name)


def _segment_bank(sp, ap, f0, source: str) -> VoiceBank:
    """Cut an utterance into vowel nuclei, voiced consonants and fricatives.

    Nuclei are energy peaks inside voiced runs (within 4 dB of the peak, at
    most +/-40ms); voiced consonants are the dips between two nuclei (nasals,
    liquids); fricatives are short unvoiced runs that are not silence.
    """
    energy = 10 * np.log10(sp.sum(axis=1) + 1e-20)
    smooth = np.convolve(energy, np.ones(5) / 5, mode='same')
    top = float(np.percentile(smooth, 99))
    nuclei, vcons, fric = [], [], []
    for s, e, is_voiced in _runs(f0 > 0):
        seg = smooth[s:e]
        if is_voiced and e - s >= 8:
            peaks: List[int] = []
            for i in np.argsort(-seg):
                if seg[i] < top - 20:
                    break
                if all(abs(int(i) - p) >= 16 for p in peaks):
                    peaks.append(int(i))
            peaks.sort()
            for p in peaks:
                lo = p
                while lo > 0 and p - lo < 8 and seg[lo - 1] >= seg[p] - 4:
                    lo -= 1
                hi = p
                while hi < len(seg) - 1 and hi - p < 8 and seg[hi + 1] >= seg[p] - 4:
                    hi += 1
                if hi - lo + 1 >= 5:
                    nuclei.append((s + lo, s + hi + 1))
            for a, b in zip(peaks[:-1], peaks[1:]):
                m = a + int(np.argmin(seg[a:b + 1]))
                if min(seg[a], seg[b]) - seg[m] >= 3.0:
                    vcons.append((s + max(0, m - 4), s + min(len(seg), m + 5)))
        elif not is_voiced and 4 <= e - s <= 80 and seg.max() > top - 24:
            k = int(np.argmax(seg))
            lo = max(0, k - 12)
            fric.append((s + lo, s + min(len(seg), lo + 24)))
    if not nuclei:
        raise ValueError(f"{source}: no vowel nuclei found")
    return VoiceBank(sp=sp, ap=ap, f0=f0, nuclei=nuclei, voiced_cons=vcons,
                     fricatives=fric, source=source)


# Peterson & Barney-style (F, bandwidth) pairs for an adult male.
_FORMANTS = {
    'a': ((730, 90), (1090, 110), (2440, 160), (3400, 250)),
    'e': ((530, 70), (1840, 100), (2480, 160), (3500, 250)),
    'i': ((300, 60), (2250, 100), (3000, 170), (3600, 250)),
    'o': ((570, 80), (840, 100), (2410, 160), (3400, 250)),
    'u': ((320, 70), (870, 100), (2240, 160), (3400, 250)),
}


def formant_bank() -> VoiceBank:
    """Source-filter fallback: analytic vowel envelopes, no recording needed."""
    n_bins = 513  # pyworld's CheapTrick FFT size at 16kHz is 1024
    f = np.linspace(0, WORLD_SR / 2, n_bins)
    tilt = 1.0 / (1.0 + (f / 500.0) ** 2)

    def env(formants):
        h = sum(1.0 / (1.0 + ((f - F) / (B / 2.0)) ** 2) for F, B in formants)
        return (h + 1e-3) * tilt

    periodic = np.clip(0.6 * (f / (WORLD_SR / 2)) ** 2, 1e-3, 1.0)
    sp_rows, ap_rows, nuclei = [], [], []
    for formants in _FORMANTS.values():
        e = env(formants)
        nuclei.append((len(sp_rows), len(sp_rows) + 24))
        for i in range(24):
            sp_rows.append(e * (1 + 0.03 * np.sin(i / 3.0)))
            ap_rows.append(periodic)
    vcons = [(len(sp_rows), len(sp_rows) + 10)]
    nasal = env(((250, 60), (1100, 200), (2300, 300))) * np.where(f > 600, 0.1, 1.0)
    sp_rows += [nasal] * 10
    ap_rows += [periodic] * 10
    fric = [(len(sp_rows), len(sp_rows) + 20)]
    sp_rows += [0.05 * np.clip(f / 4000.0, 0, 1.5) ** 4 + 1e-4] * 20
    ap_rows += [np.ones(n_bins)] * 20
    sp = np.array(sp_rows)
    return VoiceBank(sp=sp, ap=np.array(ap_rows), f0=np.zeros(len(sp)),
                     nuclei=nuclei, voiced_cons=vcons, fricatives=fric,
                     source='formant-model')


# Where the real voice comes from, for the messages that need to say so.
# Nothing is fetched automatically: the voice was once pulled out of pysptk's
# sdist with `pip download`, which runs the package's build code, unpinned.
VOICE_HINT = ("pip install pysptk (its CMU ARCTIC example is the voice), or set "
              "MELOSCRIBE_VOICE_WAV to a clean solo voice recording")


def _voice_recording(data_dir: Path) -> Optional[Path]:
    """Find the voice: MELOSCRIBE_VOICE_WAV, pysptk's CMU ARCTIC example, or
    a copy of it left under <data-dir>/voices."""
    env = os.environ.get('MELOSCRIBE_VOICE_WAV')
    if env and Path(env).exists():
        return Path(env)
    try:
        from pysptk.util import example_audio_file  # type: ignore
        p = Path(example_audio_file())
        if p.exists():
            return p
    except Exception:
        pass
    cached = Path(data_dir) / 'voices' / 'arctic_a0007.wav'
    return cached if cached.exists() else None


def load_bank(voice: str = 'auto', data_dir: Path = DEFAULT_DATA_DIR) -> VoiceBank:
    if voice == 'formant':
        return formant_bank()
    path = Path(voice) if voice not in ('auto', 'world') else _voice_recording(data_dir)
    if path is None or not path.exists():
        if voice == 'world':
            raise FileNotFoundError(f'No voice recording found: {VOICE_HINT}')
        warnings.warn('No real voice recording found: falling back to the analytic '
                      f'formant voice, which is far less realistic. For the real '
                      f'voice: {VOICE_HINT}.')
        return formant_bank()
    return analyse_recording(path)


# --------------------------------------------------------------------------
# The performance: what the singer does with the score
# --------------------------------------------------------------------------

@dataclass
class VoiceStyle:
    alpha: float = 1.0                  # formant scale (vocal tract length)
    vib_rate: float = 5.5               # Hz
    vib_cents: float = 40.0             # peak excursion
    vib_delay: float = 0.3              # s before vibrato starts on a long note
    vib_min_note: float = 0.35          # notes shorter than this get none
    vib_am_db: float = 1.5              # tremolo that rides along with vibrato
    scoop_prob: float = 0.3
    scoop_cents: Tuple[float, float] = (80.0, 200.0)
    glide_s: Tuple[float, float] = (0.05, 0.10)
    drift_cents: float = 8.0
    intonation_cents: float = 10.0
    jitter_cents: float = 3.0
    breathiness: float = 0.08
    breathy_onset: float = 0.0
    attack_s: Tuple[float, float] = (0.02, 0.05)
    release_s: Tuple[float, float] = (0.04, 0.09)
    fall_prob: float = 0.0
    shimmer_db: float = 0.8


@dataclass
class SNote:
    onset: float
    offset: float
    midi: int


@dataclass
class Syllable:
    notes: List[SNote]
    cons: str = 'none'      # none | unvoiced | voiced | glottal | h
    cons_s: float = 0.0
    vowel: int = 0
    unit: int = 0
    legato: bool = False    # voicing runs on into the next syllable
    level_db: float = 0.0


def _pre_voicing(syl: Syllable) -> float:
    """Time before the syllable's voicing that the voice is not voiced."""
    return syl.cons_s if syl.cons in ('unvoiced', 'h', 'glottal') else 0.0


def _fill(log_sp, ap, a, b, bank, u0, u1, rng, norm='frame', wander=False):
    """Copy bank frames [u0, u1) onto output frames [a, b).

    A vowel is stretched by a slow random walk through its real frames rather
    than by freezing one spectrum, which keeps the micro-variation of a real
    sustained vowel without making it periodic.
    """
    length, n_u = b - a, u1 - u0
    if length <= 0 or n_u <= 0:
        return
    if wander and length > n_u and n_u > 1:
        pos = (n_u - 1) / 2 + np.cumsum(rng.uniform(-0.35, 0.35, length))
        period = 2 * (n_u - 1)
        m = np.mod(pos, period)
        pos = np.where(m <= n_u - 1, m, period - m)
    else:
        pos = np.linspace(0, n_u - 1, length)
    idx = u0 + np.clip(np.round(pos).astype(int), 0, n_u - 1)
    scale = bank.frame_power[idx][:, None] if norm == 'frame' else bank.ref_power
    log_sp[a:b] = np.log(bank.sp[idx] / scale + 1e-30)
    ap[a:b] = bank.ap[idx]


def _fill_speech(bank, log_sp, ap, base, voiced, gain, a, b, rng, f0_scale):
    """Spoken/rapped material: the recording's own words, prosody and f0."""
    src = np.flatnonzero(bank.f0 > 0)
    pos = a
    if src.size < 40:  # formant fallback: fabricate speech-like syllables
        ref = 69 + 12 * np.log2(125.0 * f0_scale / 440.0)
        while pos < b - 30:
            length = int(rng.uniform(0.12, 0.22) / FRAME_S)
            u0, u1 = bank.nuclei[int(rng.integers(len(bank.nuclei)))]
            _fill(log_sp, ap, pos, pos + length, bank, u0, u1, rng)
            base[pos:pos + length] = (ref + rng.normal(0, 1.5)
                                      + np.linspace(1, -1, length) * rng.uniform(0.5, 2))
            voiced[pos:pos + length] = True
            gain[pos:pos + length] = np.sin(np.pi * np.arange(length) / length)
            pos += length + int(rng.uniform(0.03, 0.12) / FRAME_S)
        return
    s0, s1 = max(0, src[0] - 16), min(len(bank.f0), src[-1] + 16)
    while pos < b - 40:
        rate = rng.uniform(1.1, 1.35)          # rap is faster than read speech
        length = min(int((s1 - s0) / rate), b - pos)
        idx = s0 + np.minimum((np.arange(length) * rate).astype(int), s1 - s0 - 1)
        log_sp[pos:pos + length] = np.log(bank.sp[idx] / bank.ref_power + 1e-30)
        ap[pos:pos + length] = bank.ap[idx]
        f0 = bank.f0[idx] * f0_scale * 2 ** (rng.normal(0, 1.0) / 12)
        v = f0 > 0
        voiced[pos:pos + length] = v
        base[pos:pos + length] = np.where(v, 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440), 0)
        gain[pos:pos + length] = 1.0
        pos += length + int(rng.uniform(0.12, 0.3) / FRAME_S)


def render_voice(bank: VoiceBank, sylls: Sequence[Syllable], style: VoiceStyle,
                 duration: float, rng, speech: Sequence[Tuple[float, float]] = (),
                 speech_f0_scale: float = 1.0, trace: Optional[List[Dict]] = None):
    """Sing a line with WORLD. Returns (audio at WORLD_SR, melody f0 per 5ms
    frame with 0 where no melody is sung, the truth notes).

    `trace`, when given, receives one dict per truth note describing how it
    was sung (onset consonant, legato connection, scoop, vibrato, melisma
    position), so errors can be attributed to what the singer did. Recording
    it draws no random numbers, so it never changes the audio."""
    import pyworld as pw
    from scipy.ndimage import uniform_filter1d

    bank = bank.warped(style.alpha)
    n = int(math.ceil(duration / FRAME_S)) + 1
    log_sp = np.full((n, bank.n_bins), -60.0)
    ap = np.ones((n, bank.n_bins))
    base, mod = np.zeros(n), np.zeros(n)
    voiced, melody = np.zeros(n, bool), np.zeros(n, bool)
    gain = np.zeros(n)
    breath = np.full(n, style.breathiness)
    notes: List[Note] = []

    def fr(sec: float) -> int:
        return int(round(sec / FRAME_S))

    connected = [False] + [sylls[k - 1].legato and sylls[k].cons in ('none', 'voiced')
                           for k in range(1, len(sylls))]

    for k, syl in enumerate(sylls):
        a = max(1, fr(syl.notes[0].onset))
        b = min(n - 1, fr(syl.notes[-1].offset))
        if b - a < 4:
            continue
        lvl = 10 ** (syl.level_db / 20)
        cons_n = max(0, fr(syl.cons_s))

        # 1. what comes before voicing: a fricative, an aspiration, or a stop
        if syl.cons in ('unvoiced', 'h') and cons_n > 0:
            c = max(0, a - cons_n)
            if syl.cons == 'unvoiced' and bank.fricatives:
                u0, u1 = bank.fricatives[syl.unit % len(bank.fricatives)]
                _fill(log_sp, ap, c, a, bank, u0, u1, rng, norm='ref')
                gain[c:a] = lvl * np.minimum(_ramp(a - c) * 3, 1.0)
            else:  # 'h': the coming vowel, whispered
                u0, u1 = bank.nuclei[syl.vowel % len(bank.nuclei)]
                _fill(log_sp, ap, c, a, bank, u0, u0 + 3, rng)
                ap[c:a] = 1.0
                gain[c:a] = 0.3 * lvl * _ramp(a - c)

        # 2. a voiced consonant opens the voicing; 3. the vowel fills the rest
        v = a
        if syl.cons == 'voiced' and bank.voiced_cons and cons_n > 0:
            v = min(b - 3, a + cons_n)
            u0, u1 = bank.voiced_cons[syl.unit % len(bank.voiced_cons)]
            _fill(log_sp, ap, a, v, bank, u0, u1, rng, norm='ref')
        u0, u1 = bank.nuclei[syl.vowel % len(bank.nuclei)]
        _fill(log_sp, ap, v, b, bank, u0, u1, rng, wander=True)
        voiced[a:b] = True
        melody[a:b] = True

        # 4. level: attack, re-articulation dip, release
        g = np.full(b - a, lvl)
        if v > a:
            g[:v - a] *= 0.5
        if not connected[k]:
            att = min(b - a, max(1, fr(rng.uniform(*style.attack_s))))
            g[:att] *= _ramp(att)
            if style.breathy_onset > 0:
                L = min(b - a, fr(0.18))
                breath[a:a + L] = np.maximum(breath[a:a + L],
                                             style.breathy_onset * (1 - _ramp(L)))
        elif syl.cons == 'none':
            L = min(b - a, 5)
            g[:L] *= 0.3 + 0.7 * _ramp(L)
            p0 = max(0, a - 4)
            gain[p0:a] *= 1 - 0.7 * _ramp(a - p0)
        nxt_connected = k + 1 < len(sylls) and connected[k + 1]
        if not nxt_connected:
            rel_s = 0.015 if syl.legato else rng.uniform(*style.release_s)
            rel = min(b - a, max(1, fr(rel_s)))
            g[-rel:] *= 1 - _ramp(rel)
        gain[a:b] = g

        # 5. pitch: intended note + intonation offset, then vibrato
        first_traced = len(trace) if trace is not None else 0
        for j, note in enumerate(syl.notes):
            s = a if j == 0 else fr(note.onset)
            e = b if j == len(syl.notes) - 1 else fr(syl.notes[j + 1].onset)
            if e <= s:
                continue
            ic = style.intonation_cents
            off = float(np.clip(rng.normal(0, ic), -2 * ic, 2 * ic)) if ic > 0 else 0.0
            base[s:e] = note.midi + off / 100
            dur = (e - s) * FRAME_S
            if style.vib_cents > 0 and dur >= style.vib_min_note:
                tt = np.arange(e - s) * FRAME_S
                delay = min(style.vib_delay * rng.uniform(0.8, 1.2), 0.45 * dur)
                ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip((tt - delay) / 0.2, 0, 1))
                rate = (style.vib_rate * rng.uniform(0.94, 1.06)
                        * (1 + 0.04 * np.sin(2 * np.pi * 0.6 * tt + rng.uniform(0, 6.28))))
                phase = 2 * np.pi * np.cumsum(rate) * FRAME_S + rng.uniform(0, 6.28)
                depth = style.vib_cents / 100 * rng.uniform(0.85, 1.15)
                mod[s:e] += depth * ramp * np.sin(phase)
                gain[s:e] *= 10 ** (style.vib_am_db / 20 * ramp * np.sin(phase - 0.8))
            notes.append(Note(onset=s * FRAME_S, offset=e * FRAME_S, midi=float(note.midi)))
            if trace is not None:
                trace.append({
                    'syllable': k, 'pos': j, 'n_in_syllable': len(syl.notes),
                    'cons': syl.cons if j == 0 else 'melisma',
                    'cons_s': round(syl.cons_s, 3) if j == 0 else 0.0,
                    'connected': bool(connected[k]) if j == 0 else True,
                    'vibrato': bool(style.vib_cents > 0 and dur >= style.vib_min_note),
                    'scoop_cents': 0.0, 'level_db': round(syl.level_db, 1)})

        if not connected[k] and rng.random() < style.scoop_prob:
            depth = rng.uniform(*style.scoop_cents) / 100
            tau = rng.uniform(0.022, 0.04)
            L = min(b - a, fr(5 * tau))
            mod[a:a + L] -= depth * np.exp(-np.arange(L) * FRAME_S / tau)
            if trace is not None and len(trace) > first_traced:
                trace[first_traced]['scoop_cents'] = round(depth * 100, 1)
        if not syl.legato and style.fall_prob > 0 and rng.random() < style.fall_prob:
            L = min(b - a, fr(rng.uniform(0.07, 0.13)))
            mod[b - L:b] -= rng.uniform(0.8, 2.0) * np.linspace(0, 1, L) ** 2

    # 6. glides: between notes of a syllable, and across connected syllables
    for k, syl in enumerate(sylls):
        bounds = [x.onset for x in syl.notes[1:]]
        if connected[k]:
            bounds.append(syl.notes[0].onset)
        for c_s in bounds:
            half = max(1, fr(rng.uniform(*style.glide_s)) // 2)
            lo, hi = fr(c_s) - half, fr(c_s) + half
            if lo < 1 or hi >= n or not voiced[lo - 1:hi + 1].all():
                continue
            p0, p1 = base[lo - 1], base[hi]
            base[lo:hi] = p0 + (p1 - p0) * _ramp(hi - lo)

    for s0, s1 in speech:
        _fill_speech(bank, log_sp, ap, base, voiced, gain, fr(s0), fr(s1), rng,
                     speech_f0_scale)

    midi = (base + mod + _smooth_noise(rng, n, 60) * style.drift_cents / 100
            + _smooth_noise(rng, n, 2) * style.jitter_cents / 100)
    gain = gain * 10 ** (style.shimmer_db / 20 * _smooth_noise(rng, n, 4))
    log_sp = uniform_filter1d(log_sp, size=5, axis=0)   # coarticulation
    f0 = np.where(voiced, _hz(midi), 0.0)
    sp = np.exp(log_sp) * (np.maximum(gain, 1e-5) ** 2)[:, None]
    apf = np.clip(1 - (1 - ap) * (1 - np.clip(breath, 0, 1))[:, None], 1e-3, 1.0)
    audio = pw.synthesize(np.ascontiguousarray(f0), np.ascontiguousarray(sp),
                          np.ascontiguousarray(apf), WORLD_SR, FRAME_S * 1000)
    return audio, np.where(melody & voiced, f0, 0.0), notes


# --------------------------------------------------------------------------
# The score: melodies, phrasing, syllables
# --------------------------------------------------------------------------

MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10)
PROGRESSIONS = {'major': [(0, 'maj'), (7, 'maj'), (9, 'min'), (5, 'maj')],
                'minor': [(0, 'min'), (8, 'maj'), (3, 'maj'), (10, 'maj')]}
CHORDS = {'maj': (0, 4, 7), 'min': (0, 3, 7)}

# (beat, length) per 4/4 bar.
RHYTHMS = {
    'ballad': [[(0, 1.5), (1.5, 0.5), (2, 2)], [(0, 1), (1, 1), (2, 1), (3, 1)],
               [(0, 3), (3, 1)], [(0, 2), (2, 1.5), (3.5, 0.5)], [(0, 4)]],
    'pop': [[(0, .5), (.5, .5), (1, .5), (1.5, 1), (2.5, .5), (3, 1)],
            [(0, .75), (.75, .75), (1.5, .5), (2, .5), (2.5, .5), (3, .5), (3.5, .5)],
            [(.5, .5), (1, .5), (1.5, .5), (2, 1), (3, .5), (3.5, .5)],
            [(0, .5), (.5, .5), (1, 1.5), (2.5, 1.5)]],
    'melisma': [[(0, 1), (1, .25), (1.25, .25), (1.5, .25), (1.75, .25), (2, 2)],
                [(0, .5), (.5, .25), (.75, .25), (1, .25), (1.25, .25), (1.5, .25),
                 (1.75, .25), (2, 1), (3, 1)],
                [(0, 1.5), (1.5, .25), (1.75, .25), (2, .25), (2.25, .25), (2.5, .25),
                 (2.75, .25), (3, 1)]],
    'leaps': [[(0, 1), (1, 1), (2, 2)], [(0, 1.5), (1.5, .5), (2, 2)], [(0, 2), (2, 2)],
              [(0, 1), (1, .5), (1.5, .5), (2, 2)]],
    # Fast syllabic singing: one syllable per 16th, each with its consonant.
    'patter': [[(0, .25), (.25, .25), (.5, .25), (.75, .25), (1, .5), (1.5, .5), (2, .25),
                (2.25, .25), (2.5, .5), (3, 1)],
               [(0, .5), (.5, .25), (.75, .25), (1, .25), (1.25, .25), (1.5, .5), (2, .5),
                (2.5, .25), (2.75, .25), (3, .5), (3.5, .5)],
               [(0, .25), (.25, .25), (.5, .5), (1, .25), (1.25, .25), (1.5, .25), (1.75, .25),
                (2, 1), (3, .25), (3.25, .25), (3.5, .5)]],
}
_CONS_S = {'unvoiced': (0.05, 0.11), 'h': (0.06, 0.12), 'voiced': (0.04, 0.07),
           'glottal': (0.02, 0.035), 'none': (0.0, 0.0)}


def _scale_notes(key: int, mode: str, lo: int, hi: int) -> List[int]:
    pcs = MAJOR if mode == 'major' else MINOR
    return [m for m in range(lo, hi + 1) if (m - key) % 12 in pcs]


def _chord(case: Dict, bar: int) -> List[int]:
    root, quality = PROGRESSIONS[case['mode']][bar % 4]
    return [(case['key'] + root + i) % 12 for i in CHORDS[quality]]


def _near(pc: int, target: int) -> int:
    return min((m for m in range(target - 6, target + 7) if m % 12 == pc),
               key=lambda m: abs(m - target))


def compose_line(case, rng, start_s, bars, first_bar, register, rhythm,
                 repeat, leap) -> List[List[Tuple[float, float, int]]]:
    """Phrases of (onset_s, dur_s, midi), two bars each, a beat of breath at
    the end of every phrase, chord tones favoured on strong beats."""
    beat = 60.0 / case['bpm']
    allowed = _scale_notes(case['key'], case['mode'], *register)
    idx = len(allowed) // 2
    direction = 1
    phrases = []
    for p0 in range(0, bars, 2):
        nb = min(2, bars - p0)
        events = []
        for bi in range(nb):
            pattern = RHYTHMS[rhythm][int(rng.integers(len(RHYTHMS[rhythm])))]
            events += [((p0 + bi) * 4 + b, d, first_bar + p0 + bi) for b, d in pattern]
        end_beat = (p0 + nb) * 4 - 1.0
        events = [e for e in events if e[0] <= end_beat - 0.25]
        if not events:
            continue
        on, d, bar = events[-1]
        events[-1] = (on, min(d, end_beat - on), bar)
        phrase: List[Tuple[float, float, int]] = []
        for on, d, bar in events:
            r = rng.random()
            repeated = bool(phrase) and r < repeat
            if repeated:
                pass
            elif r < repeat + leap:
                idx += int(rng.choice([-1, 1])) * int(rng.integers(4, 8))
            elif rhythm == 'melisma' and d <= 0.25:
                if rng.random() < 0.25:
                    direction = -direction
                idx += direction
            else:
                idx += int(rng.choice([-2, -1, 1, 2], p=[.15, .35, .35, .15]))
            if idx < 0 or idx >= len(allowed):
                idx = int(np.clip(-idx if idx < 0 else 2 * (len(allowed) - 1) - idx,
                                  0, len(allowed) - 1))
                direction = -direction
            if not repeated and on % 2 == 0 and rng.random() < 0.6:
                chord = _chord(case, bar)
                cands = [i for i, m in enumerate(allowed) if m % 12 in chord]
                if cands:
                    idx = min(cands, key=lambda i: abs(i - idx))
            phrase.append((start_s + on * beat, d * beat, allowed[idx]))
        phrases.append(phrase)
    return phrases


def syllabify(case, phrases, rng, n_vowels) -> List[Syllable]:
    beat = 60.0 / case['bpm']
    names = list(case['cons'])
    probs = np.array([case['cons'][c] for c in names], dtype=float)
    probs /= probs.sum()
    out: List[Syllable] = []
    for phrase in phrases:
        groups: List[List[SNote]] = []
        for on, d, m in phrase:
            note = SNote(on, on + d, m)
            prev = groups[-1][-1] if groups else None
            join = prev is not None and prev.midi != m and (
                (case['rhythm'] == 'melisma' and d <= 0.26 * beat)
                or rng.random() < case.get('slur', 0.0))
            if join:
                groups[-1].append(note)
            else:
                groups.append([note])
        dyn = rng.uniform(-2.0, 1.0)
        for gi, group in enumerate(groups):
            cons = str(rng.choice(names, p=probs))
            if gi == 0 and cons in ('none', 'glottal'):
                cons = 'none'
            last = gi == len(groups) - 1
            out.append(Syllable(
                notes=group, cons=cons, cons_s=float(rng.uniform(*_CONS_S[cons])),
                vowel=int(rng.integers(n_vowels)), unit=int(rng.integers(1000)),
                legato=(not last) and rng.random() < case.get('legato', 0.7),
                level_db=dyn + gi * 0.15 + float(rng.normal(0, 1.0))))
    return out


def humanise(sylls: List[Syllable], rng, sd_s: float = 0.015) -> None:
    """Timing jitter, then make every boundary physically consistent: a
    consonant must fit before its vowel, a detached note must leave a gap."""
    for syl in sylls:
        shift = float(np.clip(rng.normal(0, sd_s), -2.5 * sd_s, 2.5 * sd_s))
        for j, note in enumerate(syl.notes):
            inner = 0.0 if j == 0 else float(np.clip(rng.normal(0, sd_s / 2), -sd_s, sd_s))
            note.onset += shift + inner
        for j, note in enumerate(syl.notes):
            note.offset = (syl.notes[j + 1].onset if j + 1 < len(syl.notes)
                           else note.offset + shift)
    for k in range(len(sylls) - 1):
        cur, nxt = sylls[k], sylls[k + 1]
        last = cur.notes[-1]
        room = nxt.notes[0].onset - last.onset
        nxt.cons_s = min(nxt.cons_s, 0.4 * room)
        pre = _pre_voicing(nxt)
        if cur.legato:
            last.offset = nxt.notes[0].onset - pre
        else:
            gap = rng.uniform(0.06, 0.15)
            last.offset = min(last.offset, nxt.notes[0].onset - pre - gap)
            if last.offset - last.onset < 0.08:
                cur.legato = True
                last.offset = nxt.notes[0].onset - pre


def _diatonic(m: int, steps: int, allowed: List[int]) -> int:
    i = min(range(len(allowed)), key=lambda i: abs(allowed[i] - m))
    return allowed[int(np.clip(i + steps, 0, len(allowed) - 1))]


@dataclass
class Plan:
    duration: float
    lead: List[Syllable]
    speech: List[Tuple[float, float]]
    backing: List[Tuple[List[Syllable], float, float]]   # (line, alpha scale, dB)
    inst_notes: List[Tuple[float, float, int]]
    bars: List[Tuple[int, float, str]]
    regions: List[Tuple[float, float, str]]


def compose_case(case: Dict, n_vowels: int, rng) -> Plan:
    beat = 60.0 / case['bpm']
    bar_s = 4 * beat
    t = case.get('lead_in', 0.3)
    bars, regions, speech, inst, sing = [], [], [], [], []
    bar_idx = 0
    for kind, nb in case['sections']:
        bars += [(bar_idx + i, t + i * bar_s, kind) for i in range(nb)]
        end = t + nb * bar_s
        if kind == 'sing':
            sing.append((t, nb, bar_idx))
        elif kind == 'rap':
            speech.append((t + 0.1, end - 0.25))
            regions.append((t, end, 'rap'))
        elif kind == 'inst':
            for ph in compose_line(case, rng, t, nb, bar_idx,
                                   case.get('inst_register', (60, 76)),
                                   case.get('inst_rhythm', 'pop'), 0.1, 0.15):
                inst += ph
            regions.append((t, end, 'instrumental'))
        elif kind == 'intro':
            regions.append((t, end, 'intro'))
        t, bar_idx = end, bar_idx + nb
    duration = t + case.get('tail', 1.2)

    lead: List[Syllable] = []
    for start, nb, b0 in sing:
        phrases = compose_line(case, rng, start, nb, b0, case['register'],
                               case['rhythm'], case.get('repeat', 0.1),
                               case.get('leap', 0.1))
        lead += syllabify(case, phrases, rng, n_vowels)
    humanise(lead, rng, case.get('timing_s', 0.015))

    backing: List[Tuple[List[Syllable], float, float]] = []
    kind = case.get('backing')
    if kind:
        sung = [b for b in bars if b[2] == 'sing']
        split = sung[len(sung) // 2][1] if sung else 0.0
        allowed = _scale_notes(case['key'], case['mode'], case['register'][0] - 12,
                               case['register'][1] + 12)
        if kind in ('thirds', 'both'):
            harm = []
            for syl in lead:
                if syl.notes[0].onset < split:
                    continue
                d = float(rng.normal(0, 0.012))
                harm.append(Syllable([SNote(x.onset + d, x.offset + d,
                                            _diatonic(x.midi, 2, allowed))
                                      for x in syl.notes],
                                     cons=syl.cons, cons_s=syl.cons_s, vowel=syl.vowel,
                                     unit=syl.unit, legato=syl.legato,
                                     level_db=syl.level_db))
            backing.append((harm, 1.06, -7.0))
        if kind in ('oohs', 'both'):
            for vi, (target, alpha) in enumerate(((case['register'][0] + 2, 0.97),
                                                  (case['register'][0] + 7, 1.08))):
                line = []
                for bar, t0, section in sung:
                    if t0 >= split:
                        continue
                    pcs = _chord(case, bar)
                    m = min((_near(pc, target) for pc in pcs), key=lambda x: abs(x - target))
                    line.append(Syllable([SNote(t0 + 0.04, t0 + bar_s - 0.2, m)],
                                         cons='none', vowel=vi))
                backing.append((line, alpha, -10.0))
        if kind in ('octave_below', 'thirds_below'):
            # A double through the whole song: an octave down (the classic
            # octave-error trap) or a loud third below (a second melody the
            # tracker may prefer).
            line = []
            for syl in lead:
                d = float(rng.normal(0, 0.010))
                line.append(Syllable(
                    [SNote(x.onset + d, x.offset + d,
                           x.midi - 12 if kind == 'octave_below' else _diatonic(x.midi, -2, allowed))
                     for x in syl.notes],
                    cons=syl.cons, cons_s=syl.cons_s, vowel=syl.vowel, unit=syl.unit,
                    legato=syl.legato, level_db=syl.level_db))
            backing.append((line, 0.88 if kind == 'octave_below' else 0.97,
                            case.get('backing_db', -6.0 if kind == 'octave_below' else -4.0)))
    return Plan(duration=duration, lead=lead, speech=speech, backing=backing,
                inst_notes=inst, bars=bars, regions=regions)


# --------------------------------------------------------------------------
# The band
# --------------------------------------------------------------------------

def _env(n, attack, release):
    e = np.ones(n)
    a = min(n, max(1, int(attack * SR)))
    r = min(n - a, max(1, int(release * SR)))
    e[:a] = np.linspace(0, 1, a)
    if r > 0:
        e[n - r:] *= np.linspace(1, 0, r)
    return e


def _tone(f, dur, amps, taus, rng, attack=0.005, release=0.05, detune=0.0):
    n = max(1, int(dur * SR))
    t = np.arange(n) / SR
    out = np.zeros(n)
    for h, (a, tau) in enumerate(zip(amps, taus), 1):
        fh = f * h * 2 ** (detune / 1200)
        if fh >= 0.45 * SR:
            break
        part = np.sin(2 * np.pi * fh * t + rng.uniform(0, 2 * np.pi))
        out += a * (part * np.exp(-t / tau) if tau else part)
    return out * _env(n, attack, release)


def _piano(f, d, rng):
    return _tone(f, d, [1, .5, .3, .18, .1, .06], [1.2, .8, .5, .35, .25, .2], rng, .003, .08)


def _ep(f, d, rng):
    return _tone(f, d, [1, .25, .08, .05], [1.6, .7, .4, .3], rng, .004, .1)


def _pad(f, d, rng):
    amps = [(0.8 ** h) / h for h in range(1, 11)]
    return sum(_tone(f, d, amps, [0] * 10, rng, .3, .4, det) for det in (-8, 0, 8)) / 3


def _bass(f, d, rng):
    return _tone(f, d, [1, .5, .25, .12], [.9, .5, .3, .2], rng, .004, .03)


def _chug(f, d, rng):
    x = _tone(f, d, [1 / h for h in range(1, 15)], [.25] * 14, rng, .002, .02)
    return np.tanh(2 * x) / np.tanh(2)


def _kick(rng):
    t = np.arange(int(.35 * SR)) / SR
    y = np.sin(2 * np.pi * np.cumsum(48 + 110 * np.exp(-t / .035)) / SR) * np.exp(-t / .16)
    y[:80] += .3 * rng.standard_normal(80)
    return y


def _snare(rng):
    t = np.arange(int(.25 * SR)) / SR
    hp = np.diff(rng.standard_normal(len(t)), prepend=0.0)
    return .6 * hp * np.exp(-t / .09) + .5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / .045)


def _hat(rng, open_=False):
    t = np.arange(int((.2 if open_ else .06) * SR)) / SR
    hp = np.diff(rng.standard_normal(len(t) + 2), n=2)
    return .35 * hp * np.exp(-t / (.08 if open_ else .018))


def _synth_line(notes, n, rng):
    """A saw lead with delayed vibrato: the most voice-like thing in a band."""
    out = np.zeros(n)
    for on, d, m in notes:
        length = int(d * SR)
        i = int(on * SR)
        if i + length > n or length <= 0:
            continue
        t = np.arange(length) / SR
        vib = 0.3 * np.clip((t - 0.2) / 0.15, 0, 1) * np.sin(2 * np.pi * 5.8 * t)
        phase = 2 * np.pi * np.cumsum(_hz(m + vib)) / SR
        y = sum((0.85 ** h) / h * np.sin(h * phase) for h in range(1, 15)
                if _hz(m) * h < 0.45 * SR)
        out[i:i + length] += y * _env(length, .012, .05)
    return out


def _guitar_line(notes, n, rng):
    """Karplus-Strong plucks, run as one IIR comb per note."""
    from scipy.signal import lfilter
    out = np.zeros(n)
    for on, d, m in notes:
        length = int((d + 0.05) * SR)
        i = int(on * SR)
        if i + length > n:
            continue
        period = max(2, int(round(SR / float(_hz(m)))))
        x = np.zeros(length)
        x[:period] = np.convolve(rng.uniform(-1, 1, period), [.5, .5], 'same')
        a = np.zeros(period + 2)
        a[0], a[period], a[period + 1] = 1.0, -0.4985, -0.4985
        y = lfilter([1.0], a, x) * _env(length, .002, .05)
        out[i:i + length] += np.tanh(1.5 * y)
    return out


def render_band(case: Dict, plan: Plan, n: int, rng) -> np.ndarray:
    out = np.zeros(n + SR)
    beat = 60.0 / case['bpm']
    style = case['band']
    kick, snare, hat, hat_open = _kick(rng), _snare(rng), _hat(rng), _hat(rng, True)

    def add(sig, t0, g):
        i = int(round(t0 * SR))
        if i < len(out):
            j = min(len(out), i + len(sig))
            out[i:j] += g * sig[:j - i]

    for bar, t0, _ in plan.bars:
        chord = _chord(case, bar)
        tones = sorted(_near(pc, 62) for pc in chord)
        bass = _near(chord[0], case.get('bass_centre', 41))
        if style == 'ballad':
            for m in tones:
                add(_pad(_hz(m - 12), 4 * beat, rng), t0, .10)
            arp = tones + [tones[0] + 12]
            for i in range(8):
                add(_piano(_hz(arp[i % len(arp)]), beat * 1.2, rng), t0 + i * beat / 2, .2)
            add(_bass(_hz(bass), 3.8 * beat, rng), t0, .35)
            add(kick, t0, .5)
            add(snare, t0 + 2 * beat, .25)
            for i in range(8):
                add(hat, t0 + i * beat / 2, .07)
        elif style == 'rnb':
            ep = tones + [_near((chord[0] + (10 if case['mode'] == 'minor' else 11)) % 12, 64)]
            for off in (0, 2):
                for m in ep:
                    add(_ep(_hz(m), 1.9 * beat, rng), t0 + off * beat, .12)
            for off, d in ((0, .7), (1.5, .4), (2, .7), (3.5, .4)):
                add(_bass(_hz(bass), d * beat, rng), t0 + off * beat, .33)
            for off in (0, 1.75, 2.5):
                add(kick, t0 + off * beat, .5)
            for off in (1, 3):
                add(snare, t0 + off * beat, .38)
            for i in range(16):
                add(hat, t0 + i * beat / 4, .1 if i % 2 == 0 else .06)
        elif style == 'rock':
            for i in range(8):
                for m in (_near(chord[0], 45), _near(chord[0], 45) + 7):
                    add(_chug(_hz(m), beat * .48, rng), t0 + i * beat / 2, .16)
                add(_bass(_hz(bass), beat * .45, rng), t0 + i * beat / 2, .3)
                add(hat_open if i == 7 else hat, t0 + i * beat / 2, .12)
            for off in (0, 2, 2.5):
                add(kick, t0 + off * beat, .55)
            for off in (1, 3):
                add(snare, t0 + off * beat, .45)
        else:  # pop
            for off in (.5, 1.5, 2.5, 3.5):
                for m in tones:
                    add(_piano(_hz(m), beat * .45, rng), t0 + off * beat, .13)
            for m in tones:
                add(_pad(_hz(m - 12), 4 * beat, rng), t0, .06)
            for i in range(8):
                add(_bass(_hz(bass + (12 if i % 4 == 3 else 0)), beat * .45, rng),
                    t0 + i * beat / 2, .32)
                add(hat, t0 + i * beat / 2, .14 if i % 2 == 0 else .1)
            for off in (0, 1.5, 2):
                add(kick, t0 + off * beat, .55)
            for off in (1, 3):
                add(snare, t0 + off * beat, .4)
    out = out[:n]
    if plan.inst_notes:
        render = _guitar_line if case.get('lead_inst') == 'guitar' else _synth_line
        line = render(plan.inst_notes, n, rng)
        on = np.abs(line) > 1e-4
        if on.any():
            out = out + line * 1.2 * _rms(out, on) / _rms(line, on)
    return out


def _rms(x, mask=None) -> float:
    v = x[mask] if mask is not None and np.any(mask) else x
    return float(np.sqrt(np.mean(v ** 2)) + 1e-12)


def _reverb(x, t60, wet_db, rng, predelay=0.02):
    from scipy.signal import fftconvolve, lfilter
    t = np.arange(int(t60 * SR)) / SR
    ir = lfilter([0.35], [1, -0.65], rng.standard_normal(len(t)) * np.exp(-6.91 * t / t60))
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
    wet = fftconvolve(x, ir)[:len(x)]
    d = int(predelay * SR)
    wet = np.concatenate([np.zeros(d), wet[:len(x) - d]])
    return x + 10 ** (wet_db / 20) * wet


# --------------------------------------------------------------------------
# Separation paths
# --------------------------------------------------------------------------

def proxy_separation(mix, vocals, accomp, residual_db, rng) -> np.ndarray:
    """A separator stand-in: the MP3 mix through an ideal ratio mask that is
    smeared (5 bins x 3 frames) and floored, so accompaniment leaks through
    at a level wandering inside `residual_db` and vocal partials buried under
    the band are partly lost - the two things real separators get wrong."""
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft

    n = len(vocals)
    kw = dict(fs=SR, nperseg=2048, noverlap=1536)
    X = stft(mix[:n], **kw)[2]
    pv = np.abs(stft(vocals, **kw)[2]) ** 2
    pa = np.abs(stft(accomp, **kw)[2]) ** 2
    irm = uniform_filter(pv / (pv + pa + 1e-12), size=(5, 3), mode='nearest')
    walk = _smooth_noise(rng, irm.shape[1], 80)
    walk = (walk - walk.min()) / (np.ptp(walk) + 1e-12)
    floor = 10 ** ((residual_db[0] + (residual_db[1] - residual_db[0]) * walk) / 20)
    y = istft((floor[None, :] + (1 - floor[None, :]) * irm) * X, **kw)[1]
    return np.pad(y, (0, max(0, n - len(y))))[:n]


def repet_separation(mix_path, out_path) -> Path:
    """librosa's REPET-SIM (nn_filter) vocal separation: a weak, real
    separator run on the MP3 itself - the pessimistic floor."""
    import librosa
    import soundfile as sf
    y, sr = librosa.load(str(mix_path), sr=22050, mono=True)
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(
        S_full, aggregate=np.median, metric='cosine',
        width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)
    mask = librosa.util.softmask(S_full - S_filter, 10 * S_filter, power=2)
    sf.write(str(out_path), librosa.istft(mask * S_full * phase, length=len(y)), sr)
    return Path(out_path)


def demucs_separation(mix_path, out_path) -> Path:
    """Path (iv): the real separator, where Demucs and its weights exist."""
    from ..stems import separate
    shutil.copy(separate(mix_path, preset='balanced').vocals, out_path)
    return Path(out_path)


SEPARATORS = {'repet': repet_separation, 'demucs': demucs_separation}


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

_CONS_POP = {'unvoiced': .4, 'voiced': .3, 'glottal': .15, 'none': .15}

CASES: List[Dict] = [
    # Legato ballad, wide late vibrato, portamento: the held-note case.
    dict(name='ballad_vibrato_f', bpm=70, key=62, mode='major', register=(62, 76),
         rhythm='ballad', band='ballad', sections=[('intro', 1), ('sing', 8)],
         legato=0.85, repeat=0.08, leap=0.1, slur=0.25,
         cons={'unvoiced': .3, 'voiced': .35, 'none': .2, 'h': .15},
         voice=dict(alpha=1.16, vib_rate=5.2, vib_cents=70, vib_delay=0.25,
                    scoop_prob=0.35, glide_s=(0.08, 0.14), breathiness=0.12),
         var_db=2.0, residual_db=(-22, -14), reverb=1.4),
    # Syncopated pop with many repeated, re-articulated notes.
    dict(name='pop_repeats_m', bpm=104, key=57, mode='minor', register=(52, 67),
         rhythm='pop', band='pop', sections=[('intro', 1), ('sing', 12)],
         legato=0.6, repeat=0.45, leap=0.08, slur=0.05, cons=_CONS_POP,
         voice=dict(vib_rate=5.8, vib_cents=25, vib_min_note=0.4, scoop_prob=0.25,
                    glide_s=(0.04, 0.07), fall_prob=0.3),
         var_db=1.0, residual_db=(-18, -12), reverb=0.9),
    # R&B melismatic runs: 16ths on one vowel, pitch is the only boundary cue.
    dict(name='melisma_runs_f', bpm=76, key=60, mode='minor', register=(60, 77),
         rhythm='melisma', band='rnb', sections=[('intro', 1), ('sing', 8)],
         legato=0.8, repeat=0.02, leap=0.05,
         cons={'unvoiced': .35, 'voiced': .35, 'none': .3},
         voice=dict(alpha=1.12, vib_rate=5.6, vib_cents=45, scoop_prob=0.2,
                    glide_s=(0.02, 0.04)),
         var_db=2.0, residual_db=(-20, -14), reverb=1.1),
    # Breathy, soft onsets and wide leaps.
    dict(name='breathy_leaps_f', bpm=74, key=64, mode='major', register=(55, 79),
         rhythm='leaps', band='ballad', sections=[('intro', 1), ('sing', 8)],
         legato=0.5, repeat=0.05, leap=0.45,
         cons={'h': .45, 'unvoiced': .25, 'voiced': .15, 'none': .15},
         voice=dict(alpha=1.1, vib_rate=4.8, vib_cents=55, vib_delay=0.4,
                    breathiness=0.35, breathy_onset=0.8, attack_s=(0.08, 0.16),
                    scoop_prob=0.4, scoop_cents=(100, 250)),
         var_db=2.5, residual_db=(-22, -15), reverb=1.3),
    # A rapped verse (must yield ~no notes) then a sung hook.
    dict(name='rap_then_hook_m', bpm=92, key=55, mode='minor', register=(50, 64),
         rhythm='pop', band='pop', sections=[('intro', 1), ('rap', 5), ('sing', 6)],
         legato=0.6, repeat=0.3, leap=0.1, cons=_CONS_POP,
         voice=dict(vib_cents=30, scoop_prob=0.3, fall_prob=0.2),
         var_db=1.0, residual_db=(-18, -12), reverb=0.9),
    # Instrumental break with a synth lead in the vocal register.
    dict(name='break_synth_f', bpm=100, key=65, mode='major', register=(60, 74),
         rhythm='pop', band='pop', lead_inst='synth', inst_register=(62, 77),
         sections=[('intro', 1), ('sing', 4), ('inst', 4), ('sing', 3)],
         legato=0.65, repeat=0.2, leap=0.1, cons=_CONS_POP,
         voice=dict(alpha=1.14, vib_cents=35, vib_rate=5.9),
         var_db=1.5, residual_db=(-20, -12), reverb=1.0),
    # Instrumental break with a guitar lead over a rock band.
    dict(name='break_guitar_m', bpm=96, key=52, mode='minor', register=(50, 64),
         rhythm='ballad', band='rock', lead_inst='guitar', inst_register=(57, 74),
         sections=[('intro', 1), ('sing', 4), ('inst', 4), ('sing', 3)],
         legato=0.7, repeat=0.15, leap=0.1, cons=_CONS_POP,
         voice=dict(vib_cents=35, scoop_prob=0.3),
         var_db=1.0, residual_db=(-18, -12), reverb=0.9),
    # Backing vocals: "ooh" pads in the verse, a third above in the chorus.
    dict(name='harmonies_m', bpm=90, key=57, mode='major', register=(52, 66),
         rhythm='pop', band='pop', backing='both', sections=[('intro', 1), ('sing', 10)],
         legato=0.65, repeat=0.25, leap=0.08, cons=_CONS_POP,
         voice=dict(vib_cents=30, scoop_prob=0.25),
         var_db=1.0, residual_db=(-20, -13), reverb=1.1),
    # Bass-baritone, down among the bass guitar.
    dict(name='low_male', bpm=80, key=45, mode='minor', register=(40, 55),
         rhythm='ballad', band='ballad', bass_centre=38, sections=[('intro', 1), ('sing', 8)],
         legato=0.7, repeat=0.1, leap=0.12, slur=0.15,
         cons={'unvoiced': .35, 'voiced': .35, 'none': .2, 'h': .1},
         voice=dict(alpha=0.92, vib_rate=5.0, vib_cents=35, scoop_prob=0.3),
         var_db=1.0, residual_db=(-20, -12), reverb=1.1),
    # Soprano: sparse harmonics far above the formants.
    dict(name='high_female', bpm=86, key=69, mode='major', register=(69, 86),
         rhythm='ballad', band='ballad', sections=[('intro', 1), ('sing', 9)],
         legato=0.7, repeat=0.08, leap=0.2, slur=0.2,
         cons={'unvoiced': .3, 'voiced': .3, 'none': .25, 'h': .15},
         voice=dict(alpha=1.22, vib_rate=6.0, vib_cents=55, scoop_prob=0.3),
         var_db=2.0, residual_db=(-22, -14), reverb=1.3),
    # Long held notes with fast, deep vibrato and strong tremolo: the shape
    # that once shredded held notes at the vibrato rate (HANDOFF, decision 4).
    dict(name='sustained_vibrato_m', bpm=60, key=55, mode='major', register=(55, 67),
         rhythm='ballad', band='ballad', sections=[('intro', 1), ('sing', 6)],
         legato=0.8, repeat=0.05, leap=0.1, slur=0.2,
         cons={'unvoiced': .3, 'voiced': .35, 'none': .25, 'h': .1},
         voice=dict(vib_rate=6.2, vib_cents=60, vib_delay=0.2, vib_am_db=3.0, scoop_prob=0.3),
         var_db=2.0, residual_db=(-22, -14), reverb=1.5),
    # Fast syllabic 16ths, detached: short notes cut by consonants.
    dict(name='patter_16ths_f', bpm=112, key=62, mode='major', register=(62, 74),
         rhythm='patter', band='pop', sections=[('intro', 1), ('sing', 9)],
         legato=0.3, repeat=0.3, leap=0.05,
         cons={'unvoiced': .5, 'voiced': .35, 'glottal': .15},
         voice=dict(alpha=1.12, vib_cents=20, vib_min_note=0.45, scoop_prob=0.15,
                    attack_s=(0.01, 0.03), release_s=(0.02, 0.05)),
         var_db=1.5, residual_db=(-20, -12), reverb=0.8),
    # Octave-and-more leaps sung legato, with portamento and deep scoops.
    dict(name='octave_leaps_m', bpm=84, key=50, mode='major', register=(48, 67),
         rhythm='leaps', band='rnb', sections=[('intro', 1), ('sing', 7)],
         legato=0.8, repeat=0.05, leap=0.6,
         cons={'unvoiced': .3, 'voiced': .3, 'none': .4},
         voice=dict(vib_cents=40, scoop_prob=0.45, scoop_cents=(100, 300),
                    glide_s=(0.08, 0.16)),
         var_db=1.5, residual_db=(-20, -13), reverb=1.1),
    # A spoken (female) intro before the singing starts.
    dict(name='spoken_intro_f', bpm=96, key=64, mode='minor', register=(60, 74),
         rhythm='pop', band='pop', speech_f0_scale=1.7,
         sections=[('intro', 1), ('rap', 3), ('sing', 5)],
         legato=0.6, repeat=0.25, leap=0.1, cons=_CONS_POP,
         voice=dict(alpha=1.15, vib_cents=35, scoop_prob=0.3),
         var_db=1.0, residual_db=(-18, -12), reverb=1.0),
    # The lead doubled an octave below: the production trick that invites
    # octave errors.
    dict(name='octave_double_f', bpm=96, key=62, mode='major', register=(62, 74),
         rhythm='pop', band='pop', backing='octave_below', backing_db=-6.0,
         sections=[('intro', 1), ('sing', 8)],
         legato=0.65, repeat=0.25, leap=0.08, cons=_CONS_POP,
         voice=dict(alpha=1.12, vib_cents=30, scoop_prob=0.25),
         var_db=1.0, residual_db=(-20, -13), reverb=1.0),
    # A loud harmony a third below, throughout.
    dict(name='harmony_below_f', bpm=78, key=65, mode='major', register=(65, 77),
         rhythm='ballad', band='ballad', backing='thirds_below', backing_db=-4.0,
         sections=[('intro', 1), ('sing', 7)],
         legato=0.75, repeat=0.1, leap=0.1, slur=0.2,
         cons={'unvoiced': .3, 'voiced': .35, 'none': .2, 'h': .15},
         voice=dict(alpha=1.16, vib_cents=45, scoop_prob=0.3),
         var_db=2.0, residual_db=(-22, -14), reverb=1.3),
    # Quiet, soft, drenched in reverb, and badly separated: the real-track
    # conditions under which HANDOFF measured 42% vocal coverage.
    dict(name='quiet_reverb_m', bpm=88, key=52, mode='minor', register=(50, 64),
         rhythm='pop', band='pop', sections=[('intro', 1), ('sing', 7)],
         legato=0.6, repeat=0.2, leap=0.1, cons=_CONS_POP,
         voice=dict(vib_cents=30, breathiness=0.2, attack_s=(0.06, 0.12), scoop_prob=0.3),
         var_db=-3.0, residual_db=(-15, -10), reverb=2.2, reverb_wet_db=-7.0),
    # Close-miked and dry, with audible inhalations before every phrase.
    dict(name='breaths_close_m', bpm=76, key=53, mode='major', register=(50, 65),
         rhythm='ballad', band='ballad', breaths_db=-14.0,
         sections=[('intro', 1), ('sing', 7)], legato=0.5, repeat=0.1, leap=0.1, slur=0.15,
         cons={'h': .3, 'unvoiced': .3, 'voiced': .2, 'none': .2},
         voice=dict(vib_cents=35, breathiness=0.15, breathy_onset=0.5, scoop_prob=0.3),
         var_db=2.0, residual_db=(-22, -14), reverb=0.5, reverb_wet_db=-18.0),
]

# Three seconds, for the smoke test.
SMOKE_CASE = dict(name='smoke_3s', bpm=120, key=60, mode='major', register=(60, 72),
                  rhythm='pop', band='pop', sections=[('sing', 1)], lead_in=0.4,
                  tail=0.6, legato=0.5, repeat=0.2, leap=0.1,
                  cons={'unvoiced': .4, 'voiced': .3, 'none': .3},
                  voice=dict(vib_cents=30), var_db=3.0, residual_db=(-20, -14),
                  reverb=0.6)


# --------------------------------------------------------------------------
# Truth, generation
# --------------------------------------------------------------------------

@dataclass
class Truth:
    name: str
    duration: float
    notes: List[Note]
    times: np.ndarray
    freqs: np.ndarray
    regions: List[Tuple[float, float, str]]
    case: Dict
    source: str = ''
    version: int = GENERATOR_VERSION
    case_dir: Optional[Path] = None
    note_info: List[Dict] = field(default_factory=list)   # aligned with notes

    def audio(self, path: str = 'clean') -> Path:
        return self.case_dir / {'clean': 'vocals.wav', 'mix': 'mix.mp3'}.get(path, f'{path}.wav')

    def ground_truth(self) -> GroundTruth:
        return GroundTruth(name=self.name, times=self.times, freqs=self.freqs,
                           notes=list(self.notes), meta={'realistic': True})

    def save(self, path: Path) -> None:
        # Atomic: parallel scorers read these files while generation may run.
        tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps({
            'name': self.name, 'duration': self.duration, 'version': self.version,
            'source': self.source, 'case': self.case, 'regions': self.regions,
            'notes': [[round(n.onset, 4), round(n.offset, 4), n.midi] for n in self.notes],
            'note_info': self.note_info,
            'f0_hop': DEFAULT_HOP, 'f0': [round(float(f), 3) for f in self.freqs],
        }), encoding='utf-8')
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> 'Truth':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        freqs = np.asarray(d['f0'], dtype=float)
        return cls(name=d['name'], duration=d['duration'],
                   notes=[Note(on, off, float(m)) for on, off, m in d['notes']],
                   times=np.arange(len(freqs)) * d['f0_hop'], freqs=freqs,
                   regions=[tuple(r) for r in d['regions']], case=d['case'],
                   source=d.get('source', ''), version=d.get('version', 0),
                   case_dir=Path(path).parent, note_info=d.get('note_info', []))


def _write_mp3(path: Path, audio: np.ndarray) -> None:
    import soundfile as sf
    # libsndfile's LAME binding maps compression level linearly onto
    # 320..32 kbps for MPEG-1; this lands on 128 kbps CBR. Decoding it back
    # through the same library was measured sample-aligned (lag 0).
    sf.write(str(path), audio, SR, format='MP3', subtype='MPEG_LAYER_III',
             compression_level=(320 - MP3_KBPS) / 288, bitrate_mode='CONSTANT')


def generate_case(case: Dict, data_dir: Path, bank: Optional[VoiceBank] = None,
                  voice: str = 'auto', force: bool = False) -> Truth:
    import soundfile as sf
    from scipy.signal import resample_poly

    data_dir = Path(data_dir)
    case_dir = data_dir / case['name']
    truth_path = case_dir / 'truth.json'
    if truth_path.exists() and not force:
        cached = Truth.load(truth_path)
        # Sung by another voice - say the formant fallback, before the real
        # voice was installed - it is stale, or it would stand in for it.
        same_voice = bank is None or cached.source == bank.source
        if (cached.version == GENERATOR_VERSION and same_voice
                and cached.case == json.loads(json.dumps(case))):
            if cached.notes and not cached.note_info:
                # Written before annotations existed: re-sing the lead only
                # (deterministic) to recover them, leaving the audio alone.
                notes, trace = _lead_trace(case, bank or load_bank(voice, data_dir))
                # Stored times are rounded to 0.1ms, so compare with tolerance.
                if len(notes) == len(cached.notes) and all(
                        abs(a.onset - b.onset) < 1e-3 and a.midi == b.midi
                        for a, b in zip(notes, cached.notes)):
                    cached.note_info = trace
                    cached.save(truth_path)
                else:
                    warnings.warn(f"{case['name']}: cannot recover note annotations "
                                  f"(different voice?); regenerate with --force")
            return cached
    case_dir.mkdir(parents=True, exist_ok=True)
    bank = bank or load_bank(voice, data_dir)
    rng = np.random.default_rng(_seed(case['name'], GENERATOR_VERSION))

    plan = compose_case(case, len(bank.nuclei), rng)
    style = VoiceStyle(**case.get('voice', {}))
    n = int(plan.duration * SR)
    trace: List[Dict] = []

    def to_sr(y):
        y = resample_poly(y, 441, 160)
        return np.pad(y, (0, max(0, n - len(y))))[:n]

    def sample_mask(f0_frames):
        idx = np.minimum(np.round(np.arange(n) / (SR * FRAME_S)).astype(int),
                         len(f0_frames) - 1)
        return f0_frames[idx] > 0

    lead16, melody_f0, notes = render_voice(
        bank, plan.lead, style, plan.duration, rng, speech=plan.speech,
        speech_f0_scale=case.get('speech_f0_scale', 1.0), trace=trace)
    sung = sample_mask(melody_f0)
    lead = to_sr(lead16)
    lead *= 0.1 / _rms(lead, sung)
    backing = np.zeros(n)
    for line, alpha, level_db in plan.backing:
        bstyle = replace(style, alpha=style.alpha * alpha, vib_cents=style.vib_cents * 0.5,
                         scoop_prob=0.1, breathiness=style.breathiness + 0.05)
        y16, bf0, _ = render_voice(bank, line, bstyle, plan.duration, rng)
        y = to_sr(y16)
        backing += y * (0.1 / _rms(y, sample_mask(bf0))) * 10 ** (level_db / 20)
    if case.get('breaths_db') is not None:
        # Inhalations before phrases, from their own random stream so that
        # enabling them never changes anything else about a case.
        backing += _breaths(plan.lead, n, case['breaths_db'],
                            np.random.default_rng(_seed(case['name'], 'breaths')))
    band = render_band(case, plan, n, rng)
    band *= 0.1 * 10 ** (-case.get('var_db', 1.0) / 20) / _rms(band, sung)

    vocals = _reverb(lead + backing, case.get('reverb', 1.0), case.get('reverb_wet_db', -13.0), rng)
    accomp = _reverb(band, 0.7, -17.0, rng)
    mix = vocals + accomp
    scale = 0.89 / (np.max(np.abs(mix)) + 1e-12)
    vocals, accomp, mix = vocals * scale, accomp * scale, mix * scale

    sf.write(str(case_dir / 'vocals.wav'), vocals, SR, subtype='PCM_16')
    sf.write(str(case_dir / 'accomp.wav'), accomp, SR, subtype='PCM_16')
    _write_mp3(case_dir / 'mix.mp3', mix)
    decoded, _ = sf.read(str(case_dir / 'mix.mp3'), always_2d=True)
    proxy = proxy_separation(decoded.mean(axis=1), vocals, accomp,
                             case.get('residual_db', (-24, -12)), rng)
    sf.write(str(case_dir / 'proxy.wav'), proxy, SR, subtype='PCM_16')
    for stale in ('repet.wav', 'demucs.wav'):
        (case_dir / stale).unlink(missing_ok=True)

    hop_ratio = int(round(DEFAULT_HOP / FRAME_S))
    n_grid = int(math.ceil(plan.duration / DEFAULT_HOP))
    freqs = np.zeros(n_grid)
    sub = melody_f0[::hop_ratio][:n_grid]
    freqs[:len(sub)] = sub
    truth = Truth(name=case['name'], duration=n / SR, notes=notes,
                  times=np.arange(n_grid) * DEFAULT_HOP, freqs=freqs,
                  regions=[(round(a, 4), round(b, 4), k) for a, b, k in plan.regions],
                  case=json.loads(json.dumps(case)), source=bank.source,
                  case_dir=case_dir, note_info=trace)
    truth.save(truth_path)
    return truth


def _breaths(lead: Sequence[Syllable], n: int, level_db: float, rng) -> np.ndarray:
    """Band-limited inhalation noise before every phrase that follows a rest,
    ending just before the phrase's first consonant. Unvoiced, so absent from
    the truth: anything transcribed here is spurious. Level is relative to the
    lead's sung RMS (0.1 after normalisation)."""
    from scipy.signal import butter, sosfilt
    sos = butter(2, [350 / (SR / 2), 5500 / (SR / 2)], btype='band', output='sos')
    out = np.zeros(n)
    prev_end = -1.0
    for syl in lead:
        start = syl.notes[0].onset - _pre_voicing(syl)
        if start - prev_end >= 0.45:
            dur = min(rng.uniform(0.25, 0.45), start - prev_end - 0.15)
            i0 = int((start - 0.05 - dur) * SR)
            length = int(dur * SR)
            if i0 > 0 and length > 0:
                y = sosfilt(sos, rng.standard_normal(length))
                y *= np.sin(np.pi * np.arange(length) / length) ** 1.5
                out[i0:i0 + length] += 0.1 * 10 ** (level_db / 20) * y / _rms(y)
        prev_end = syl.notes[-1].offset
    return out


def _lead_trace(case: Dict, bank: VoiceBank) -> Tuple[List[Note], List[Dict]]:
    """Re-sing a case's lead exactly as `generate_case` did (same seed, same
    draw order) and return its notes with their annotations."""
    rng = np.random.default_rng(_seed(case['name'], GENERATOR_VERSION))
    plan = compose_case(case, len(bank.nuclei), rng)
    trace: List[Dict] = []
    _, _, notes = render_voice(bank, plan.lead, VoiceStyle(**case.get('voice', {})),
                               plan.duration, rng, speech=plan.speech,
                               speech_f0_scale=case.get('speech_f0_scale', 1.0), trace=trace)
    return notes, trace


def ensure_path_audio(truth: Truth, path: str) -> Path:
    """The audio a system should see for this evaluation path, made lazily."""
    target = truth.audio(path)
    if path in SEPARATORS and not target.exists():
        # Written aside and moved into place: an interrupted separation must
        # never leave a partial file that later looks like a finished one.
        partial = target.with_name(f'{target.stem}.partial.wav')
        SEPARATORS[path](truth.audio('mix'), partial)
        os.replace(partial, target)
    if not target.exists():
        raise FileNotFoundError(f"{truth.name}: no audio for path {path!r} ({target})")
    return target


def load_truths(data_dir: Path, names: Optional[Sequence[str]] = None) -> List[Truth]:
    data_dir = Path(data_dir)
    order = [c['name'] for c in CASES]
    found = {p.parent.name: p for p in data_dir.glob('*/truth.json')}
    wanted = list(names) if names else [n for n in order if n in found]
    missing = [w for w in wanted if w not in found]
    if missing:
        raise FileNotFoundError(f"No generated case(s) {missing} under {data_dir}; "
                                f"run `generate` first")
    return [Truth.load(found[w]) for w in wanted]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _intervals(notes: Sequence[Note]):
    iv = np.array([[n.onset, n.offset] for n in notes], dtype=float).reshape(-1, 2)
    return iv, np.array([n.midi for n in notes], dtype=float)


def _pitch_label(d: float) -> str:
    if abs(d) <= 0.5:
        return 'ok'
    k = round(d / 12)
    if k != 0 and abs(d - 12 * k) <= 0.5:
        return 'octave'
    return 'semitone' if abs(d) <= 1.5 else 'wrong_pitch'


def _region_at(t: float, regions) -> str:
    for a, b, kind in regions:
        if a <= t < b:
            return kind
    return 'gap'


def diagnose(ref_iv, ref_midi, est_iv, est_midi, matches, regions) -> Counter:
    """Name the cause of every missed reference note (fn:*) and every
    unmatched detection (fp:*), so an F1 gap can be attributed."""
    counts: Counter = Counter()
    m_ref = {r for r, _ in matches}
    m_est = {e for _, e in matches}

    def overlap(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    for r, (on, off) in enumerate(ref_iv):
        if r in m_ref:
            continue
        cands = [e for e in range(len(est_iv)) if overlap(on, off, *est_iv[e]) > 0]
        if not cands:
            counts['fn:missed'] += 1
            continue
        at = [e for e in cands if est_iv[e][0] <= on + 0.03 < est_iv[e][1]]
        e = at[0] if at else max(cands, key=lambda i: overlap(on, off, *est_iv[i]))
        label = _pitch_label(est_midi[e] - ref_midi[r])
        if label == 'ok':
            if est_iv[e][0] < on - ONSET_TOL:
                label = 'merged' if e in m_est else 'early_onset'
            elif est_iv[e][0] > on + ONSET_TOL:
                label = 'late_onset'
            else:
                label = 'other'
        counts['fn:' + label] += 1

    for e, (on, off) in enumerate(est_iv):
        if e in m_est:
            continue
        cands = [r for r in range(len(ref_iv)) if overlap(on, off, *ref_iv[r]) > 0]
        if not cands:
            region = _region_at((on + off) / 2, regions)
            counts['fp:' + (region if region in ('rap', 'instrumental') else 'in_gap')] += 1
            continue
        r = max(cands, key=lambda i: overlap(on, off, *ref_iv[i]))
        label = _pitch_label(est_midi[e] - ref_midi[r])
        if label == 'ok':
            if on > ref_iv[r][0] + ONSET_TOL:
                # Only the first right-pitch detection on a note is its
                # (late) onset; any later one is a piece of a shredded note.
                earlier = any(e2 != e and est_iv[e2][0] < on
                              and abs(est_midi[e2] - ref_midi[r]) <= 0.5
                              and overlap(*ref_iv[r], *est_iv[e2]) > 0
                              for e2 in range(len(est_iv)))
                label = 'fragment' if (r in m_ref or earlier) else 'late_onset'
            elif on < ref_iv[r][0] - ONSET_TOL:
                label = 'early_onset'
            else:
                label = 'other'
        counts['fp:' + label] += 1
    return counts


def score_case(truth: Truth, pred: Prediction) -> Dict:
    """Frame, note and usability metrics for one case."""
    import mir_eval
    from .groundtruth import f0_to_notes
    from .metrics import score as base_score

    base = base_score(truth.ground_truth(), pred, system='realistic')
    f, nt = base.frame, base.note
    row: Dict = {
        'OA': f.get('Overall Accuracy', 0.0), 'RPA': f.get('Raw Pitch Accuracy', 0.0),
        'RCA': f.get('Raw Chroma Accuracy', 0.0), 'VR': f.get('Voicing Recall', 0.0),
        'VFA': f.get('Voicing False Alarm', 0.0), 'oct_frames': f.get('octave_error_rate', 0.0),
        'onset_bias_ms': 1000 * nt.get('onset_bias', 0.0),
        'onset_mae_ms': 1000 * nt.get('onset_mae', 0.0),
    }

    # Pitch accuracy against the *intended* notes, where the truth is voiced:
    # separates "wrong note" from "right note, quantised through the vibrato".
    est_f = resample_f0(pred.times, pred.freqs, truth.times)
    intended = np.zeros_like(truth.freqs)
    for n in truth.notes:
        intended[(truth.times >= n.onset) & (truth.times < n.offset)] = _hz(n.midi)
    v = truth.freqs > 0
    ok = v & (est_f > 0) & (intended > 0)
    cents = np.zeros_like(est_f)
    cents[ok] = 1200 * np.abs(np.log2(est_f[ok] / intended[ok]))
    row['RPA_note'] = float(np.sum(ok & (cents <= PITCH_TOL_CENTS)) / max(1, v.sum()))

    est_notes = pred.notes if pred.notes is not None else f0_to_notes(pred.times, pred.freqs)
    ref_iv, ref_m = _intervals(truth.notes)
    est_iv, est_m = _intervals(est_notes)
    n_ref, n_est = len(ref_iv), len(est_iv)
    matches: List[Tuple[int, int]] = []
    p = r = f1 = f1_off = 0.0
    if n_ref and n_est:
        kw = dict(onset_tolerance=ONSET_TOL, pitch_tolerance=PITCH_TOL_CENTS)
        matches = mir_eval.transcription.match_notes(
            ref_iv, _hz(ref_m), est_iv, _hz(est_m), offset_ratio=None, **kw)
        p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_iv, _hz(ref_m), est_iv, _hz(est_m), offset_ratio=None, **kw)
        f1_off = mir_eval.transcription.precision_recall_f1_overlap(
            ref_iv, _hz(ref_m), est_iv, _hz(est_m), offset_ratio=0.2,
            offset_min_tolerance=0.05, **kw)[2]
    row.update(note_f1=float(f1), note_p=float(p), note_r=float(r),
               note_f1_off=float(f1_off), n_ref=n_ref, n_est=n_est,
               tp=len(matches), fp=n_est - len(matches), fn=n_ref - len(matches))

    m_est = {e for _, e in matches}
    dur = est_iv[:, 1] - est_iv[:, 0] if n_est else np.zeros(0)
    spurious = sum(1 for e in range(n_est) if e not in m_est and dur[e] < SPURIOUS_MAX_S)
    mids, gmids = est_iv.mean(axis=1), ref_iv.mean(axis=1)
    covers = [int(np.sum((mids >= a) & (mids < b))) for a, b in ref_iv]
    per_est = [int(np.sum((gmids >= a) & (gmids < b))) for a, b in est_iv]
    octave = semitone = overlapped = 0
    rap = inst = gap = 0
    for e, (a, b) in enumerate(est_iv):
        ov = np.maximum(0, np.minimum(b, ref_iv[:, 1]) - np.maximum(a, ref_iv[:, 0])) \
            if n_ref else np.zeros(0)
        if n_ref and ov.max() > 0:
            if ov.max() >= 0.5 * (b - a):
                overlapped += 1
                label = _pitch_label(est_m[e] - ref_m[int(np.argmax(ov))])
                octave += label == 'octave'
                semitone += label == 'semitone'
            continue
        kind = _region_at((a + b) / 2, truth.regions)
        rap += kind == 'rap'
        inst += kind == 'instrumental'
        gap += kind not in ('rap', 'instrumental')
    row.update(
        spurious_rate=spurious / max(1, n_est),
        fragmentation=float(np.mean([c for c in covers if c >= 1])) if any(covers) else 0.0,
        merge_rate=sum(c for c in per_est if c >= 2) / max(1, n_ref),
        octave_note_rate=octave / max(1, overlapped),
        semitone_note_rate=semitone / max(1, overlapped),
        n_in_rap=rap, n_in_inst=inst, n_in_gap=gap,
        phantom_rate=(rap + inst + gap) / max(1, n_est))
    row['errors'] = dict(diagnose(ref_iv, ref_m, est_iv, est_m, matches, truth.regions))
    return row


def oracle_prediction(truth: Truth) -> Prediction:
    return Prediction(name=truth.name, times=truth.times.copy(), freqs=truth.freqs.copy(),
                      notes=[Note(n.onset, n.offset, n.midi) for n in truth.notes])


class RealisticOracle:
    name = 'oracle'

    def __init__(self, truths: Sequence[Truth]):
        self.by_dir = {t.name: t for t in truths}

    def run(self, audio_path: Path) -> Prediction:
        return oracle_prediction(self.by_dir[Path(audio_path).parent.name])


_MEAN_KEYS = ('note_f1', 'note_f1_off', 'note_p', 'note_r', 'OA', 'RPA', 'RPA_note', 'RCA',
              'VR', 'VFA', 'oct_frames', 'spurious_rate', 'fragmentation', 'merge_rate',
              'octave_note_rate', 'semitone_note_rate', 'phantom_rate', 'onset_bias_ms',
              'onset_mae_ms')
_SUM_KEYS = ('n_ref', 'n_est', 'tp', 'fp', 'fn', 'n_in_rap', 'n_in_inst', 'n_in_gap',
             'runtime_s', 'audio_s')


def aggregate_rows(rows: Sequence[Dict]) -> Dict:
    agg = {k: float(np.mean([r[k] for r in rows])) for k in _MEAN_KEYS}
    agg.update({k: float(np.sum([r.get(k, 0) for r in rows])) for k in _SUM_KEYS})
    tp, fp, fn = agg['tp'], agg['fp'], agg['fn']
    agg['note_f1_pooled'] = 2 * tp / max(1.0, 2 * tp + fp + fn)
    agg['n_cases'] = len(rows)
    return agg


def rank_error_sources(rows: Sequence[Dict]) -> List[Dict]:
    """Pooled over cases: what each error class costs in note F1, i.e. the
    pooled F1 if every error of that class were fixed and nothing else."""
    tp = sum(r['tp'] for r in rows)
    fp = sum(r['fp'] for r in rows)
    fn = sum(r['fn'] for r in rows)
    total: Counter = Counter()
    for r in rows:
        total.update(r['errors'])
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    labels = sorted({k.split(':', 1)[1] for k in total})
    ranked = []
    for label in labels:
        a, b = total.get('fn:' + label, 0), total.get('fp:' + label, 0)
        fixed = 2 * (tp + a) / max(1, 2 * (tp + a) + (fp - b) + (fn - a))
        where = Counter()
        for r in rows:
            where[r['case']] += r['errors'].get('fn:' + label, 0) + r['errors'].get('fp:' + label, 0)
        ranked.append({'source': label, 'fn': a, 'fp': b, 'f1_if_fixed': fixed,
                       'f1_gain': fixed - f1,
                       'worst_cases': [c for c, k in where.most_common(3) if k]})
    return sorted(ranked, key=lambda x: -x['f1_gain'])


@contextmanager
def stage_timer(totals: Dict[str, float]):
    """Attribute ensemble runtime to its stages by wrapping them in place."""
    from .. import audio as audio_mod
    from ..pitch import engine as eng
    from ..pitch import voters as vt
    patched = []

    def wrap(owner, attr, label):
        orig = getattr(owner, attr)

        def inner(*a, **k):
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                totals[label] = totals.get(label, 0.0) + time.perf_counter() - t0
        setattr(owner, attr, inner)
        patched.append((owner, attr, orig))

    wrap(vt.CrepeVoter, 'observe', 'crepe')
    wrap(vt.BasicPitchVoter, 'observe', 'basic_pitch_voter')
    wrap(audio_mod, 'prepare_for_pitch', 'load_condition')
    wrap(eng, 'decode', 'fuse_decode')
    wrap(eng.PitchEngine, '_segment', 'segment')
    try:
        yield totals
    finally:
        for owner, attr, orig in reversed(patched):
            setattr(owner, attr, orig)


def make_system(name: str, truths: Sequence[Truth]):
    if name == 'oracle':
        return RealisticOracle(truths)
    from .systems import get_system
    return get_system(name)


def _code_fingerprint() -> bytes:
    """Everything a voter's output depends on besides its input audio: the
    voter, grid and conditioning code, and the model library versions. Any
    edit to those files invalidates every cached voter output."""
    from .. import audio as audio_mod
    from ..pitch import grid, voters
    h = hashlib.sha1()
    for mod in (voters, grid, audio_mod):
        h.update(Path(mod.__file__).read_bytes())
    for dist in ('torchcrepe', 'basic-pitch', 'torch', 'librosa', 'onnxruntime'):
        try:
            h.update(f"{dist}={importlib.metadata.version(dist)}".encode())
        except Exception:
            pass
    return h.digest()


@contextmanager
def voter_cache(cache_dir: Optional[Path], stats: Dict[str, float]):
    """Memoise voter outputs on disk, keyed by content, not by name.

    CREPE is ~97% of the ensemble's runtime on CPU, and tuning fusion or
    segmentation does not change what the voters report. With this, a re-run
    after such a change costs seconds per case instead of minutes, while any
    change to the voters themselves (or their audio) misses the cache.
    """
    if cache_dir is None:
        yield
        return
    from ..pitch import voters as vt
    from ..pitch.grid import VoterOutput
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _code_fingerprint()
    patched = []
    for cls in set(vt._REGISTRY.values()):
        def cached(self, audio, n_frames, _orig=cls.observe):
            h = hashlib.sha1(fingerprint)
            h.update(type(self).__name__.encode())
            h.update(repr(sorted((k, repr(v)) for k, v in vars(self).items())).encode())
            h.update(np.ascontiguousarray(audio.samples).tobytes())
            h.update(f"{audio.sr}/{n_frames}".encode())
            if audio.path is not None and Path(audio.path).exists():
                h.update(Path(audio.path).read_bytes())
            f = cache_dir / f"{type(self).__name__}_{h.hexdigest()[:24]}.npz"
            if f.exists():
                try:
                    with np.load(f, allow_pickle=False) as z:
                        meta = json.loads(str(z['meta_json']))
                        meta.update({k[5:]: z[k] for k in z.files
                                     if k.startswith('meta_') and k != 'meta_json'})
                        out = VoterOutput(name=str(z['name']), salience=z['salience'],
                                          voicing=z['voicing'], weight=float(z['weight']),
                                          meta=meta)
                    stats['voter_cache_hits'] = stats.get('voter_cache_hits', 0) + 1
                    return out
                except Exception:
                    pass
            out = _orig(self, audio, n_frames)
            arrays = {f'meta_{k}': np.asarray(v) for k, v in out.meta.items()
                      if isinstance(v, np.ndarray)}
            plain = {k: v for k, v in out.meta.items() if not isinstance(v, np.ndarray)}
            tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, name=out.name, salience=out.salience, voicing=out.voicing,
                                weight=out.weight, meta_json=json.dumps(plain, default=str),
                                **arrays)
            os.replace(tmp, f)
            stats['voter_cache_misses'] = stats.get('voter_cache_misses', 0) + 1
            return out
        patched.append((cls, cls.observe))
        cls.observe = cached
    try:
        yield
    finally:
        for cls, orig in reversed(patched):
            cls.observe = orig


def _pitch_code_hash() -> str:
    """Everything that decides a system's output (voters, fusion, engine,
    systems): a saved row is only resumed if this is unchanged."""
    from ..pitch import engine
    h = hashlib.sha1(_code_fingerprint())
    for f in sorted(Path(engine.__file__).parent.glob('*.py')):
        h.update(f.read_bytes())
    h.update(Path(__file__).with_name('systems.py').read_bytes())
    return h.hexdigest()[:16]


def _row_key(code: str, truth: 'Truth', path: str) -> str:
    """Everything a saved row was scored from: the pitch code (`code`), the
    scoring code, the case's truth and the audio the system heard on `path`.
    A row is resumed only if all of it is unchanged - checking the pitch code
    alone reused rows scored on a case since regenerated (with the real voice
    after the formant fallback, say)."""
    h = hashlib.sha1(code.encode())
    here = Path(__file__)
    for module in (here, here.with_name('metrics.py'), here.with_name('groundtruth.py')):
        h.update(module.read_bytes())
    h.update((truth.case_dir / 'truth.json').read_bytes())
    audio = truth.audio(path)
    if audio.exists():
        stat = audio.stat()
        h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


_WORKER_SYSTEMS: Dict[str, object] = {}


def _init_worker() -> None:
    # CREPE on CPU barely scales with threads (13 frames/s on 4 threads vs 11
    # on 1, measured), so one thread per process is ~3.4x the throughput.
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _score_task(task) -> Tuple[str, str, str, Dict, Dict]:
    path, sysname, case_name, data_dir, out_dir, cache_dir = task
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    truth = load_truths(Path(data_dir), [case_name])[0]
    if sysname == 'oracle':
        system = RealisticOracle([truth])
    else:
        system = _WORKER_SYSTEMS.get(sysname) or make_system(sysname, [])
        _WORKER_SYSTEMS[sysname] = system
    audio = ensure_path_audio(truth, path)
    stages: Dict[str, float] = {}
    t0 = time.perf_counter()
    with voter_cache(Path(cache_dir) if cache_dir else None, stages), stage_timer(stages), \
            redirect_stdout(io.StringIO()):
        pred = system.run(audio)
    row = score_case(truth, pred)
    row.update(case=truth.name, runtime_s=time.perf_counter() - t0, audio_s=truth.duration)
    dump = Path(out_dir) / 'predictions' / path / sysname / f'{truth.name}.json'
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(json.dumps({'notes': [[round(n.onset, 4), round(n.offset, 4), float(n.midi)]
                                          for n in (pred.notes or [])]}))
    return path, sysname, case_name, row, stages


def run_scoring(systems: Sequence[str], paths: Sequence[str], data_dir: Path,
                cases: Optional[Sequence[str]] = None, out_dir: Optional[Path] = None,
                tag: str = 'latest', workers: int = 1, cache_dir: Optional[Path] = None,
                verbose: bool = True, resume: bool = False) -> Dict:
    """Score every (path, system, case), in parallel worker processes when
    `workers` > 1. Results are identical either way; files are rewritten after
    each path so partial results are usable while later paths still run."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else data_dir / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    truths = load_truths(data_dir, cases)
    for path in paths:   # separate once, up front: workers must not race on it
        for truth in truths:
            if path in SEPARATORS and not truth.audio(path).exists():
                t0 = time.perf_counter()
                ensure_path_audio(truth, path)
                if verbose:
                    print(f"  separated {truth.name} ({path}) in {time.perf_counter() - t0:.1f}s",
                          flush=True)
    results: Dict = {'tag': tag, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                     'generator_version': GENERATOR_VERSION,
                     'voice_source': sorted({t.source for t in truths}),
                     'cases': [t.name for t in truths], 'workers': workers,
                     'voter_cache': str(cache_dir) if cache_dir else None,
                     'audio_s': round(sum(t.duration for t in truths), 1), 'paths': {}}
    durations = {t.name: t.duration for t in truths}
    code = _pitch_code_hash()
    # ProcessPoolExecutor, not multiprocessing.Pool: when a worker is killed
    # (measured here: the OOM killer, each CPU worker holds ~1.6GB), Pool
    # silently loses the task and waits forever at 0% CPU; the executor raises
    # BrokenProcessPool instead. Finished rows are saved as they arrive, so
    # `--resume` continues a crashed run without redoing them.
    from concurrent.futures import ProcessPoolExecutor, as_completed
    pool = (ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                initializer=_init_worker) if workers > 1 else None)
    try:
        for path in paths:
            started = time.perf_counter()
            tasks = sorted(((path, s, t.name, str(data_dir), str(out_dir),
                             str(cache_dir) if cache_dir else None)
                            for s in systems for t in truths),
                           key=lambda task: -durations[task[2]])
            done: Dict[Tuple[str, str], Tuple[Dict, Dict]] = {}
            keys = {t.name: _row_key(code, t, path) for t in truths}
            todo = []
            for task in tasks:
                saved = out_dir / 'rows' / path / task[1] / f'{task[2]}.json'
                if resume and saved.exists():
                    rec = json.loads(saved.read_text(encoding='utf-8'))
                    if rec.get('key') == keys[task[2]]:
                        done[(task[1], task[2])] = (rec['row'], rec['stages'])
                        continue
                todo.append(task)
            if pool:
                futures = [pool.submit(_score_task, task) for task in todo]
                stream = (f.result() for f in as_completed(futures))
            else:
                stream = map(_score_task, todo)
            for _, sysname, case_name, row, stages in stream:
                done[(sysname, case_name)] = (row, stages)
                saved = out_dir / 'rows' / path / sysname / f'{case_name}.json'
                saved.parent.mkdir(parents=True, exist_ok=True)
                tmp = saved.with_name(saved.name + '.tmp')
                tmp.write_text(json.dumps({'code': code, 'key': keys[case_name],
                                           'row': row, 'stages': stages},
                                          default=float), encoding='utf-8')
                os.replace(tmp, saved)
                if verbose:
                    print(f"  [{path}/{sysname}] {case_name:20s} F1={row['note_f1']:.3f} "
                          f"F1off={row['note_f1_off']:.3f} OA={row['OA']:.3f} "
                          f"n={row['n_est']}/{row['n_ref']} {row['runtime_s']:.1f}s", flush=True)
            results['paths'][path] = {}
            for sysname in systems:
                rows = [done[(sysname, t.name)][0] for t in truths]
                stages: Counter = Counter()
                for t in truths:
                    stages.update(done[(sysname, t.name)][1])
                results['paths'][path][sysname] = {
                    'aggregate': aggregate_rows(rows), 'cases': rows,
                    'stages_s': {k: round(v, 2) for k, v in stages.items()},
                    'error_sources': rank_error_sources(rows)}
            results['paths'][path]['_wall_s'] = round(time.perf_counter() - started, 1)
            _write_results(results, out_dir, tag)
    except Exception as exc:
        if exc.__class__.__name__ == 'BrokenProcessPool':
            print(f"\nA worker process died (most likely out of memory: each CPU worker "
                  f"holds ~1.6GB). Finished cases are saved under {out_dir / 'rows'}; rerun "
                  f"with --resume and fewer --workers.", file=sys.stderr)
        raise
    finally:
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)
    md = to_markdown(results)
    if verbose:
        print('\n' + md)
        print(f"\nwritten: {out_dir / f'realistic_{tag}.json'} (+ .md)")
    return results


def _write_results(results: Dict, out_dir: Path, tag: str) -> None:
    (out_dir / f'realistic_{tag}.json').write_text(
        json.dumps(results, indent=1, default=float), encoding='utf-8')
    (out_dir / f'realistic_{tag}.md').write_text(to_markdown(results), encoding='utf-8')


def _table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def to_markdown(results: Dict) -> str:
    lines = [f"# Realistic vocal benchmark - {results['tag']}", '',
             f"{len(results['cases'])} cases, {results['audio_s']}s of audio, voice source "
             f"{', '.join(results['voice_source'])}, generator v{results['generator_version']}, "
             f"{results['created']}", '',
             'Note F1: onset 50ms + pitch 50c (mir_eval); on+off adds offset 20%/50ms. '
             'spur = unmatched notes <150ms / detected; frag = detections per covered '
             'truth note; merge = truth notes sharing one detection; oct = octave-wrong '
             'notes / overlapping detections; phantom = detections overlapping no truth '
             'note (rap / instrumental / gap counts).', '']
    f3 = '{:.3f}'.format
    for path, systems in results['paths'].items():
        wall = systems.get('_wall_s')
        systems = {k: v for k, v in systems.items() if not k.startswith('_')}
        lines += [f"## Path: {path}", '']
        if wall is not None:
            lines += [f"wall clock {wall:.0f}s with {results.get('workers', 1)} worker(s); "
                      f"per-case seconds below are single-worker time", '']
        head = ['system', 'noteF1', 'F1 on+off', 'P', 'R', 'pooledF1', 'OA', 'RPA', 'RPA(note)',
                'RCA', 'VFA', 'spur', 'frag', 'merge', 'oct', 'phantom (rap/inst/gap)',
                'sec / audio-sec']
        rows = []
        for name, res in systems.items():
            a = res['aggregate']
            rows.append([name, f3(a['note_f1']), f3(a['note_f1_off']), f3(a['note_p']),
                         f3(a['note_r']), f3(a['note_f1_pooled']), f3(a['OA']), f3(a['RPA']),
                         f3(a['RPA_note']), f3(a['RCA']), f3(a['VFA']), f3(a['spurious_rate']),
                         f"{a['fragmentation']:.2f}", f3(a['merge_rate']),
                         f3(a['octave_note_rate']),
                         f"{int(a['n_in_rap'])}/{int(a['n_in_inst'])}/{int(a['n_in_gap'])}",
                         f"{a['runtime_s']:.0f} / {a['audio_s']:.0f}"])
        lines += [_table(head, rows), '']
        for name, res in systems.items():
            lines += [f"### {path} / {name}: per case", '']
            head = ['case', 'noteF1', 'on+off', 'P', 'R', 'OA', 'RPA', 'RPA(note)', 'VFA',
                    'spur', 'frag', 'merge', 'oct', 'rap/inst/gap', 'est/ref', 'sec']
            rows = [[r['case'], f3(r['note_f1']), f3(r['note_f1_off']), f3(r['note_p']),
                     f3(r['note_r']), f3(r['OA']), f3(r['RPA']), f3(r['RPA_note']),
                     f3(r['VFA']), f3(r['spurious_rate']), f"{r['fragmentation']:.2f}",
                     f3(r['merge_rate']), f3(r['octave_note_rate']),
                     f"{r['n_in_rap']}/{r['n_in_inst']}/{r['n_in_gap']}",
                     f"{r['n_est']}/{r['n_ref']}", f"{r['runtime_s']:.1f}"]
                    for r in res['cases']]
            lines += [_table(head, rows), '']
            if res.get('stages_s'):
                lines += ['stage seconds: ' + ', '.join(f"{k} {v}" for k, v in
                                                          sorted(res['stages_s'].items(),
                                                                 key=lambda x: -x[1])), '']
            if name != 'oracle':
                lines += [f"#### {path} / {name}: error sources (pooled F1 "
                          f"{res['aggregate']['note_f1_pooled']:.3f})", '']
                rows = [[e['source'], e['fn'], e['fp'], f3(e['f1_if_fixed']),
                         f"+{e['f1_gain']:.3f}", ', '.join(e['worst_cases'])]
                        for e in res['error_sources']]
                lines += [_table(['error source', 'missed truth notes', 'false detections',
                                  'F1 if fixed', 'gain', 'worst cases'], rows), '']
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Analysis: attribute errors to what the singer did
# --------------------------------------------------------------------------

def onset_kind(truth: Truth, i: int) -> str:
    """How truth note i begins, from the generator's annotations."""
    info = truth.note_info[i]
    if info['pos'] > 0:
        return 'melisma step (glide)'
    prev = truth.notes[i - 1] if i > 0 else None
    same = (prev is not None and prev.midi == truth.notes[i].midi
            and abs(prev.offset - truth.notes[i].onset) < 0.2)
    if info['connected']:
        return f"legato {'vowel' if info['cons'] == 'none' else 'nasal'}, " \
               f"{'same pitch' if same else 'new pitch'}"
    return {'unvoiced': 'after fricative', 'h': 'after h (breathy)',
            'glottal': 'after glottal stop', 'voiced': 'nasal after a gap',
            'none': 'vowel after a gap'}[info['cons']] + (', same pitch' if same else '')


def note_labels(truth: Truth, est_notes: Sequence[Note]):
    """Per-note outcome: 'tp' or the diagnose() cause, plus the onset error
    of the first right-pitch detection overlapping each truth note."""
    import mir_eval
    ref_iv, ref_m = _intervals(truth.notes)
    est_iv, est_m = _intervals(est_notes)
    matches = []
    if len(ref_iv) and len(est_iv):
        matches = mir_eval.transcription.match_notes(
            ref_iv, _hz(ref_m), est_iv, _hz(est_m), onset_tolerance=ONSET_TOL,
            pitch_tolerance=PITCH_TOL_CENTS, offset_ratio=None)
    labels = ['tp' if r in {x for x, _ in matches} else None for r in range(len(ref_iv))]
    # Diagnose one note at a time, but keep every detection's matched status
    # (ref index -1 never collides): 'merged' depends on it.
    matched_elsewhere = [(-1, e) for _, e in matches]
    for r in range(len(ref_iv)):
        if labels[r] is None:
            one = diagnose(ref_iv[r:r + 1], ref_m[r:r + 1], est_iv, est_m,
                           matched_elsewhere, truth.regions)
            fn = [k for k in one if k.startswith('fn:')]
            labels[r] = fn[0][3:] if fn else 'other'
    onset_err = []
    for r, (on, off) in enumerate(ref_iv):
        cands = [e for e in range(len(est_iv)) if abs(est_m[e] - ref_m[r]) <= 0.5
                 and min(off, est_iv[e][1]) > max(on, est_iv[e][0])]
        onset_err.append(min(est_iv[e][0] for e in cands) - on if cands else None)
    return labels, onset_err


def analyse(results_json: Path, data_dir: Path) -> str:
    """Break the stored predictions of a run down by onset kind, phantom
    location and fragment cause. Needs only the saved predictions."""
    res = json.loads(Path(results_json).read_text(encoding='utf-8'))
    pred_root = Path(results_json).parent / 'predictions'
    truths = {t.name: t for t in load_truths(data_dir, res['cases'])}
    lines = [f"# Error analysis - {res['tag']}", '']
    for path, systems in res['paths'].items():
        for sysname in [s for s in systems if not s.startswith('_')]:
            kinds: Dict[str, Counter] = {}
            errs: Dict[str, List[float]] = {}
            phantom = Counter()
            frag = Counter()
            for name, truth in truths.items():
                f = pred_root / path / sysname / f'{name}.json'
                if not f.exists() or not truth.note_info:
                    continue
                est = [Note(a, b, m) for a, b, m in json.loads(f.read_text())['notes']]
                labels, onset_err = note_labels(truth, est)
                for i, (lab, oe) in enumerate(zip(labels, onset_err)):
                    kind = onset_kind(truth, i)
                    kinds.setdefault(kind, Counter())[lab] += 1
                    scoop = 'scooped' if truth.note_info[i]['scoop_cents'] else None
                    if scoop:
                        kinds.setdefault('(any) scooped onset', Counter())[lab] += 1
                    if oe is not None:
                        errs.setdefault(kind, []).append(oe)
                ref_iv, ref_m = _intervals(truth.notes)
                est_iv, est_m = _intervals(est)
                for e, (a, b) in enumerate(est_iv):
                    ov = np.maximum(0, np.minimum(b, ref_iv[:, 1]) - np.maximum(a, ref_iv[:, 0]))
                    if ov.max() > 0:
                        continue
                    region = _region_at((a + b) / 2, truth.regions)
                    if region in ('rap', 'instrumental', 'intro'):
                        phantom[region] += 1
                    elif np.any((a - ref_iv[:, 1] >= -0.01) & (a - ref_iv[:, 1] < 0.4)):
                        phantom['right after a sung note (tail)'] += 1
                    elif np.any((ref_iv[:, 0] - b >= -0.01) & (ref_iv[:, 0] - b < 0.3)):
                        phantom['right before a sung note (breath/consonant)'] += 1
                    else:
                        phantom['other silence'] += 1
                # Fragments: a right-pitch detection that starts inside a truth
                # note after an earlier right-pitch detection of it.
                for r, (on, off) in enumerate(ref_iv):
                    same = sorted(e for e in range(len(est_iv)) if abs(est_m[e] - ref_m[r]) <= 0.5
                                  and min(off, est_iv[e][1]) > max(on, est_iv[e][0]))
                    for e0, e1 in zip(same[:-1], same[1:]):
                        gap = est_iv[e1][0] - est_iv[e0][1]
                        vib = truth.note_info[r]['vibrato']
                        frag[('split, no gap' if gap < 0.015 else 'voicing dropout')
                             + (' (vibrato note)' if vib else '')] += 1
            if not kinds:
                continue
            lines += [f"## {path} / {sysname}", '', '### Truth notes by onset kind', '']
            head = ['onset kind', 'n', 'found', 'late', 'missed', 'merged', 'wrong pitch',
                    'early', 'median onset err ms (first right-pitch detection)']
            rows = []
            for kind, c in sorted(kinds.items(), key=lambda kv: -sum(kv[1].values())):
                n = sum(c.values())
                wrong = c['semitone'] + c['octave'] + c['wrong_pitch']
                e = errs.get(kind, [])
                rows.append([kind, n, f"{c['tp'] / n:.2f}", f"{c['late_onset'] / n:.2f}",
                             f"{c['missed'] / n:.2f}", f"{c['merged'] / n:.2f}",
                             f"{wrong / n:.2f}", f"{c['early_onset'] / n:.2f}",
                             f"{1000 * np.median(e):+.0f} (n={len(e)})" if e else '-'])
            lines += [_table(head, rows), '', '### Phantom notes (overlap no truth note)', '']
            lines += [_table(['where', 'count'], [[k, v] for k, v in phantom.most_common()]), '']
            lines += ['### Extra right-pitch detections inside one truth note', '']
            lines += [_table(['kind', 'count'], [[k, v] for k, v in frag.most_common()]), '']
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate', help='render the cases (cached unless --force)')
    sc = sub.add_parser('score', help='score systems on generated cases')
    an = sub.add_parser('analyse', help='break a stored run down by error cause')
    an.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR))
    an.add_argument('--tag', default='latest')
    an.add_argument('--out', help='results directory (default <data-dir>/results)')
    sub.add_parser('list', help='list the cases')
    for p in (gen, sc):
        p.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR))
        p.add_argument('--cases', help='comma-separated case names')
    gen.add_argument('--force', action='store_true')
    gen.add_argument('--voice', default='auto',
                     help="'auto' (recording, else formant model), 'world', 'formant' "
                          "or a path to a voice WAV")
    sc.add_argument('--systems', default='ensemble')
    sc.add_argument('--path', default=','.join(DEFAULT_PATHS),
                    help=f"comma-separated evaluation paths from {PATHS}, or 'all'")
    sc.add_argument('--out', help='results directory (default <data-dir>/results)')
    sc.add_argument('--tag', default='latest')
    sc.add_argument('--workers', type=int, default=1,
                    help='parallel CPU worker processes (default 1). Only worth it without a '
                         'GPU; each holds ~1.6GB (CREPE full + basic-pitch)')
    sc.add_argument('--resume', action='store_true',
                    help='reuse per-case rows saved by an interrupted run with identical code')
    sc.add_argument('--voter-cache',
                    help='voter-output cache directory (default <data-dir>/voter_cache)')
    sc.add_argument('--no-voter-cache', action='store_true',
                    help='always recompute voters (use when timing the engine)')
    args = parser.parse_args(argv)

    if args.command == 'list':
        for c in CASES:
            print(f"  {c['name']:20s} {c['bpm']:>4} bpm  {c['sections']}")
        return 0

    if args.command == 'analyse':
        out = Path(args.out) if args.out else Path(args.data_dir) / 'results'
        md = analyse(out / f'realistic_{args.tag}.json', Path(args.data_dir))
        (out / f'realistic_{args.tag}_analysis.md').write_text(md, encoding='utf-8')
        print(md)
        return 0

    names = [c.strip() for c in args.cases.split(',')] if args.cases else None
    if args.command == 'generate':
        cases = [c for c in CASES if not names or c['name'] in names]
        bank = load_bank(args.voice, Path(args.data_dir))
        print(f"voice: {bank.source} ({len(bank.nuclei)} nuclei, {len(bank.voiced_cons)} "
              f"voiced consonants, {len(bank.fricatives)} fricatives)")
        for case in cases:
            t0 = time.perf_counter()
            truth = generate_case(case, Path(args.data_dir), bank=bank, force=args.force)
            durs = [n.offset - n.onset for n in truth.notes]
            print(f"  {truth.name:20s} {truth.duration:5.1f}s  {len(truth.notes):3d} notes  "
                  f"median {np.median(durs):.2f}s  midi {min(n.midi for n in truth.notes):.0f}-"
                  f"{max(n.midi for n in truth.notes):.0f}  voiced {np.mean(truth.freqs > 0):.0%}"
                  f"  ({time.perf_counter() - t0:.1f}s)", flush=True)
        return 0

    paths = PATHS[:3] if args.path == 'all' else tuple(p.strip() for p in args.path.split(','))
    bad = [p for p in paths if p not in PATHS]
    if bad:
        parser.error(f"unknown path(s) {bad}; choose from {PATHS}")
    systems = [s.strip() for s in args.systems.split(',') if s.strip()]
    cache = None if args.no_voter_cache else Path(args.voter_cache or
                                                  Path(args.data_dir) / 'voter_cache')
    run_scoring(systems, paths, Path(args.data_dir), names, args.out, args.tag,
                workers=max(1, args.workers), cache_dir=cache, resume=args.resume)
    return 0


if __name__ == '__main__':
    sys.exit(main())
