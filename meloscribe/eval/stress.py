"""Stress testing: map where note detection breaks, one axis at a time.

    python -m meloscribe.eval.stress sweep [--axis vibrato_depth,snr_db] [--systems ensemble,crepe,basic_pitch]
    python -m meloscribe.eval.stress sweep --axis all          # primary + secondary axes
    python -m meloscribe.eval.stress fuzz  [--case silence,nan_sample] [--entries engine,pipeline,cli]
    python -m meloscribe.eval.stress list

`synth.py` asks "is the easy case still perfect?". This module asks the
opposite: hold everything easy, push ONE property of the input from easy to
extreme, and record where note F1 falls through 0.9 and 0.7. The per-voter
evidence (each voter's own argmax pitch and voicing against the truth) is
captured from the same engine run, so a failure can be attributed to the stage
that gave way rather than to "the ensemble".

`crepe` and `basic_pitch` in `--systems` are the *engine* run with only that
voter: they are re-decoded from the voter outputs of the single ensemble
inference, through the engine's own decode and segmentation, so they cost no
extra inference. `bp_notes` is the raw basic-pitch note-event baseline from
`systems.py` (a separate inference).

The fuzz half runs the real entry points (`PitchEngine.transcribe`,
`Pipeline.run(vocals_only=True, lyrics_mode='off')` and the CLI) on hostile
files, and records crashes with tracebacks, hangs, and silent wrong answers.

Everything is seeded from the axis/point name via crc32 (never `hash()`, which
is randomised per process), so a rerun renders identical audio.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import signal
import sys
import time
import traceback
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .groundtruth import Note, hz_to_midi, notes_to_f0

SR = 44100
TAIL_S = 0.5
DEFAULT_OUT = Path('data/results/stress')   # gitignored via data/results/
NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')


def seed_for(*parts) -> int:
    return zlib.crc32(':'.join(str(p) for p in parts).encode('utf-8'))


def note_name(midi: float) -> str:
    m = int(round(midi))
    return f"{NAMES[m % 12]}{m // 12 - 1}"


# --------------------------------------------------------------------------
# A voice-like source
# --------------------------------------------------------------------------
#
# Source-filter: a glottal source (harmonic series falling at -6 dB/octave,
# i.e. glottal flow at -12 plus lip radiation at +6) with per-cycle jitter and
# shimmer, shaped by a parallel bank of five vowel formants that changes vowel
# on every note, plus band-passed aspiration noise pulsing with the glottal
# cycle. It is rendered additively - the formant envelope is evaluated at each
# harmonic's instantaneous frequency, which is what a formant filter does to a
# harmonic source - so nothing aliases even at C7.

VOWELS = {
    # (centre Hz, bandwidth Hz) for F1..F5; adult averages.
    'a': ((730, 90), (1090, 110), (2440, 160), (3400, 250), (4500, 300)),
    'e': ((530, 70), (1840, 120), (2480, 170), (3400, 250), (4500, 300)),
    'i': ((300, 60), (2250, 150), (3000, 200), (3600, 250), (4500, 300)),
    'o': ((570, 80), (840, 90), (2410, 160), (3400, 250), (4500, 300)),
    'u': ((330, 60), (870, 90), (2240, 160), (3400, 250), (4500, 300)),
}
FORMANT_GAINS = np.array([1.0, 0.63, 0.35, 0.22, 0.12])   # 0,-4,-9,-13,-18 dB
FORMANT_FLOOR = 0.03


@dataclass
class VoiceParams:
    vibrato_cents: float = 25.0       # peak deviation (semi-extent)
    vibrato_hz: float = 5.5
    jitter: float = 0.004             # s.d. of each glottal period, fraction
    shimmer: float = 0.04             # s.d. of each cycle's amplitude, fraction
    breath_db: float = -30.0          # aspiration noise re. voiced RMS
    tilt: float = 1.0                 # harmonic h has amplitude h**-tilt
    h1_db: float = 0.0                # extra gain on the fundamental
    h23_db: float = 0.0               # extra gain on harmonics 2 and 3
    attack_s: float = 0.03
    release_s: float = 0.05
    wander_cents: float = 3.0         # slow random pitch wander (s.d.)
    detune_cents: float = 0.0         # global, from A440
    scoop_cents: float = 0.0          # start this far flat, rise to pitch...
    scoop_s: float = 0.0              # ...over roughly this long
    glide_s: float = 0.0              # portamento from the previous note
    drift_cents: float = 0.0          # linear drift across each note
    legato_dip_db: Optional[float] = None   # abutting notes dip only this far
    vowels: Tuple[str, ...] = ('a', 'e', 'i', 'o', 'u')
    max_harmonic_hz: float = 8000.0


def synth_voice(notes: Sequence[Note], vp: VoiceParams, sr: int = SR,
                seed: int = 0, duration: Optional[float] = None) -> np.ndarray:
    """Render a monophonic sung line (unnormalised)."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(seed)
    if duration is None:
        duration = max(n.offset for n in notes) + TAIL_S
    n = int(round(duration * sr))
    t = np.arange(n) / sr

    # ---- pitch contour, in MIDI, per sample
    midi = np.full(n, np.nan)
    owner = np.full(n, -1, dtype=int)
    gains = rng.uniform(0.8, 1.0, len(notes))
    for i, note in enumerate(notes):
        s, e = int(round(note.onset * sr)), min(n, int(round(note.offset * sr)))
        if e <= s:
            continue
        local = np.arange(e - s) / sr
        seg = np.full(e - s, float(note.midi))
        if vp.scoop_cents > 0 and vp.scoop_s > 0:
            seg -= vp.scoop_cents / 100.0 * np.exp(-local / (vp.scoop_s / 3.0))
        if vp.glide_s > 0 and i > 0 and note.onset - notes[i - 1].offset < 0.15:
            w = np.clip(local / vp.glide_s, 0.0, 1.0)
            w = 0.5 - 0.5 * np.cos(np.pi * w)
            seg += (notes[i - 1].midi - note.midi) * (1.0 - w)
        if vp.drift_cents:
            seg += vp.drift_cents / 100.0 * local / max(local[-1], 1e-9)
        midi[s:e] = seg
        owner[s:e] = i
    valid = ~np.isnan(midi)
    if not valid.any():
        return np.zeros(n)
    idx = np.where(valid, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    idx[:int(np.argmax(valid))] = int(np.argmax(valid))
    midi, owner = midi[idx], owner[idx]   # hold pitch/timbre through rests

    cents = vp.detune_cents + vp.vibrato_cents * np.sin(
        2 * np.pi * vp.vibrato_hz * t + rng.uniform(0, 2 * np.pi))
    if vp.wander_cents > 0:
        knots = np.arange(0.0, duration + 0.2, 0.1)
        cents = cents + np.interp(t, knots,
                                  rng.normal(0, vp.wander_cents, knots.size))
    f0 = 440.0 * 2.0 ** ((midi - 69.0) / 12.0 + cents / 1200.0)

    if vp.jitter > 0:
        cycle = np.floor(np.cumsum(f0) / sr).astype(np.int64)
        f0 = f0 * (1.0 + vp.jitter * rng.standard_normal(cycle[-1] + 2)[cycle])
    cycles = np.cumsum(f0) / sr
    phase = 2 * np.pi * cycles

    # ---- formant envelope at a 5 ms control rate
    hop = max(1, int(sr * 0.005))
    ctrl = np.arange(0, n, hop)
    f0_c = f0[ctrl]
    table = np.array([VOWELS[v] for v in vp.vowels], dtype=float)
    formants = table[np.mod(np.maximum(owner[ctrl], 0), len(vp.vowels))]
    k = 6   # smooth vowel changes over ~30 ms
    for j in range(formants.shape[1]):
        for m in range(2):
            col = np.pad(formants[:, j, m], (k // 2, k - 1 - k // 2), mode='edge')
            formants[:, j, m] = np.convolve(col, np.ones(k) / k, mode='valid')
    F, B = formants[:, :, 0], formants[:, :, 1]

    limit = min(vp.max_harmonic_hz, 0.45 * sr)
    n_harm = int(limit / max(float(np.min(f0)), 20.0))
    samples = np.arange(n)
    voiced = np.zeros(n)
    for h in range(1, n_harm + 1):
        fh = h * f0_c
        env = FORMANT_FLOOR + (FORMANT_GAINS[None, :] / np.sqrt(
            1.0 + ((fh[:, None] - F) / (B / 2.0)) ** 2)).sum(axis=1)
        amp = env * float(h) ** (-vp.tilt)
        if h == 1:
            amp = amp * 10 ** (vp.h1_db / 20)
        elif h in (2, 3):
            amp = amp * 10 ** (vp.h23_db / 20)
        amp = np.where(fh < limit, amp, 0.0)
        if not np.any(amp):
            continue
        voiced += np.interp(samples, ctrl, amp) * np.sin(
            h * phase + rng.uniform(0, 2 * np.pi))

    if vp.shimmer > 0:
        cyc = np.floor(cycles).astype(np.int64)
        g = 1.0 + vp.shimmer * rng.standard_normal(cyc[-1] + 2)[cyc]
        w = max(1, int(sr * 0.001))
        voiced *= np.convolve(g, np.ones(w) / w, mode='same')
    voiced /= np.max(np.abs(voiced)) + 1e-12

    # ---- amplitude envelope
    env = np.zeros(n)
    for i, note in enumerate(notes):
        s, e = int(round(note.onset * sr)), min(n, int(round(note.offset * sr)))
        L = e - s
        if L <= 2:
            continue
        a = max(1, min(int(vp.attack_s * sr), L // 3))
        r = max(1, min(int(vp.release_s * sr), L // 3))
        g = np.full(L, gains[i])
        g[:a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
        g[L - r:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(r) / r)
        env[s:e] = np.maximum(env[s:e], g)
    if vp.legato_dip_db is not None:
        floor = 10 ** (vp.legato_dip_db / 20)
        for i in range(len(notes) - 1):
            prev, nxt = notes[i], notes[i + 1]
            if nxt.onset - prev.offset < 1e-3:
                s = max(0, int((prev.offset - vp.release_s) * sr))
                e = min(n, int((nxt.onset + vp.attack_s) * sr))
                env[s:e] = np.maximum(env[s:e],
                                      floor * min(gains[i], gains[i + 1]))

    if vp.breath_db > -100:
        sos = butter(2, [500 / (sr / 2), min(6000.0, 0.45 * sr) / (sr / 2)],
                     btype='band', output='sos')
        noise = sosfilt(sos, rng.standard_normal(n)) * (0.7 + 0.3 * np.cos(phase))
        loud = env > 0.5
        ref = np.sqrt(np.mean(voiced[loud] ** 2)) if loud.any() else 0.3
        noise *= ref * 10 ** (vp.breath_db / 20) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
        voiced = voiced + noise
    return voiced * env


# --------------------------------------------------------------------------
# Scenarios: one easy default, and one knob turned per axis
# --------------------------------------------------------------------------

PHRASE = [(0, 0.45), (2, 0.45), (4, 0.45), (7, 0.6), (5, 0.45), (4, 0.45),
          (2, 0.45), (0, 0.9), (4, 0.45), (7, 0.45), (9, 0.6), (7, 0.9)]
C_MAJOR = (0, 2, 4, 5, 7, 9, 11)


def phrase(base: int = 60, gap: float = 0.08, start: float = 0.3,
           quick: bool = False) -> List[Note]:
    """The easy default: 12 notes, steps and leaps up to a sixth, C-major."""
    notes, t = [], start
    for offset, dur in (PHRASE[:4] if quick else PHRASE):
        notes.append(Note(onset=t, offset=t + dur, midi=float(base + offset)))
        t += dur + gap
    return notes


def run_of(offsets: Sequence[int], base: int, dur: float, gap: float,
           start: float = 0.3) -> List[Note]:
    notes, t = [], start
    for off in offsets:
        notes.append(Note(onset=t, offset=t + dur, midi=float(base + off)))
        t += dur + gap
    return notes


def diatonic_shift(midi: float, steps: int) -> float:
    m = int(round(midi))
    octave, pc = divmod(m, 12)
    if pc not in C_MAJOR:
        return float(m - 3)
    o, d = divmod(C_MAJOR.index(pc) + steps, 7)
    return float((octave + o) * 12 + C_MAJOR[d])


@dataclass
class Scenario:
    notes: List[Note]
    voice: VoiceParams = field(default_factory=VoiceParams)
    harmony_db: Optional[float] = None       # a third below, re. lead RMS
    snr_db: Optional[float] = None           # same-register accompaniment
    unison_cents: Optional[float] = None     # detuned double at equal level
    reverb_rt60: Optional[float] = None
    clip_drive_db: Optional[float] = None
    band: Optional[Tuple[float, float]] = None
    peak_dbfs: float = -1.0
    out_sr: int = SR
    mp3_kbps: Optional[int] = None
    alt_notes: Optional[List[Note]] = None   # competing line, scored separately
    info: Dict[str, Any] = field(default_factory=dict)


def _staccato(d: float, quick: bool) -> Scenario:
    offs = [0, 2, 4, 7, 5, 4, 2, 0, 4, 7, 9, 7, 5, 4, 2, 0]
    return Scenario(run_of(offs[:6] if quick else offs, 60, d, max(d, 0.04)))


def _sixteenths(bpm: float, quick: bool) -> Scenario:
    ioi = 15.0 / bpm
    offs = [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0, 2, 4, 5, 7, 9,
            7, 5, 4, 2]
    return Scenario(run_of(offs[:8] if quick else offs, 60, 0.9 * ioi, 0.1 * ioi))


def _repeats(label: str, quick: bool) -> Scenario:
    gap_ms, _, dip = label.partition('/dip')
    gap = float(gap_ms.rstrip('ms')) / 1000.0
    k = 3 if quick else 5
    notes = run_of([4] * k, 60, 0.35, gap)
    notes += run_of([7] * k, 60, 0.35, gap, start=notes[-1].offset + 0.3)
    vp = VoiceParams(legato_dip_db=float(dip.rstrip('dB')) if dip else None)
    return Scenario(notes, vp)


def _harmony(db: float, quick: bool) -> Scenario:
    lead = phrase(quick=quick)
    alt = [Note(n.onset, n.offset, diatonic_shift(n.midi, -2)) for n in lead]
    return Scenario(lead, harmony_db=db, alt_notes=alt)


def _register(base: int, quick: bool) -> Scenario:
    return Scenario(phrase(base=base, quick=quick))


def _long_drift(cents: float, quick: bool) -> Scenario:
    L = 5.0 if quick else 20.0
    notes = [Note(0.3, 0.3 + L, 57.0), Note(0.4 + L, 0.4 + 2 * L, 62.0)]
    return Scenario(notes, VoiceParams(drift_cents=-cents))


@dataclass
class Axis:
    name: str
    description: str
    points: List[Any]                              # easy -> extreme
    build: Callable[[Any, bool], Scenario]
    label: Callable[[Any], str] = str


AXES: Dict[str, Axis] = {a.name: a for a in [
    Axis('baseline', 'the easy default, for reference', [0],
         lambda v, q: Scenario(phrase(quick=q)), lambda v: 'easy'),
    Axis('vibrato_depth', 'vibrato semi-extent in cents at 5.5 Hz',
         [0, 50, 100, 150, 200],
         lambda v, q: Scenario(phrase(quick=q), VoiceParams(vibrato_cents=v)),
         lambda v: f"+-{v}c"),
    Axis('vibrato_rate', 'vibrato rate in Hz at +-60 cents', [3, 5.5, 7, 9],
         lambda v, q: Scenario(phrase(quick=q),
                               VoiceParams(vibrato_cents=60, vibrato_hz=v)),
         lambda v: f"{v}Hz"),
    Axis('note_duration', 'staccato notes of this length, 50% duty',
         [0.4, 0.2, 0.12, 0.08, 0.06, 0.04], _staccato,
         lambda v: f"{int(v * 1000)}ms"),
    Axis('sixteenths_bpm', 'legato 16th-note scale runs at this tempo',
         [80, 120, 160, 200], _sixteenths, lambda v: f"{v}bpm"),
    Axis('repeat_gap', 'same-pitch repeats: gap, then legato dips',
         ['100ms', '50ms', '25ms', '10ms', '0ms', '0ms/dip-12dB',
          '0ms/dip-6dB'], _repeats),
    Axis('harmony_db', 'backing vocal a third below, level re. lead',
         [-20, -12, -6, 0, 3, 6], _harmony, lambda v: f"{v:+d}dB"),
    Axis('snr_db', 'piano chords in the same register: vocal SNR',
         [20, 10, 5, 0, -5],
         lambda v, q: Scenario(phrase(quick=q), snr_db=v),
         lambda v: f"{v:+d}dB"),
    Axis('h1_db', 'fundamental attenuation (H2/H3 +6 dB), E3-C#4',
         [0, -10, -20, -30, -120],
         lambda v, q: Scenario(phrase(base=52, quick=q),
                               VoiceParams(h1_db=v, h23_db=6.0 if v else 0.0)),
         lambda v: 'absent' if v <= -100 else f"{v:+d}dB"),
    Axis('register', 'phrase transposed: lowest..highest note',
         [40, 48, 60, 72, 84, 87], _register,
         lambda v: f"{note_name(v)}-{note_name(v + 9)}"),
    Axis('mp3_kbps', 'MP3 CBR bitrate (the MP3 is the input)',
         [320, 128, 64, 32],
         lambda v, q: Scenario(phrase(quick=q), mp3_kbps=v),
         lambda v: f"{v}kbps"),
    # ---- secondary: nearly free, run with --axis all
    Axis('detune_cents', 'global detune from A440', [-45, -30, -15, 15, 30, 45],
         lambda v, q: Scenario(phrase(quick=q), VoiceParams(detune_cents=v)),
         lambda v: f"{v:+d}c"),
    Axis('scoop_cents', 'scoop up into every note over 150 ms',
         [100, 200, 300],
         lambda v, q: Scenario(phrase(quick=q),
                               VoiceParams(scoop_cents=v, scoop_s=0.15)),
         lambda v: f"{v}c"),
    Axis('glide_s', 'legato portamento from the previous note',
         [0.05, 0.1, 0.2],
         lambda v, q: Scenario(phrase(gap=0.0, quick=q),
                               VoiceParams(glide_s=v, legato_dip_db=-6.0)),
         lambda v: f"{int(v * 1000)}ms"),
    Axis('unison_cents', 'equal-level double detuned by this much, 20 ms late',
         [5, 10, 15, 25],
         lambda v, q: Scenario(phrase(quick=q), unison_cents=v),
         lambda v: f"{v}c"),
    Axis('reverb_rt60', 'synthetic reverb tail, wet -3 dB', [0.3, 1.0, 2.0, 4.0],
         lambda v, q: Scenario(phrase(quick=q), reverb_rt60=v),
         lambda v: f"{v}s"),
    Axis('band', 'band-limited (telephone)', [(100, 8000), (300, 3400), (500, 2500)],
         lambda v, q: Scenario(phrase(quick=q), band=v),
         lambda v: f"{v[0]}-{v[1]}Hz"),
    Axis('clip_drive_db', 'drive into hard clipping', [6, 12, 20, 30],
         lambda v, q: Scenario(phrase(quick=q), clip_drive_db=v),
         lambda v: f"+{v}dB"),
    Axis('peak_dbfs', 'overall level (16-bit WAV)', [-20, -35, -50, -60],
         lambda v, q: Scenario(phrase(quick=q), peak_dbfs=v),
         lambda v: f"{v}dBFS"),
    Axis('sample_rate', 'file sample rate', [8000, 16000, 22050, 48000, 96000],
         lambda v, q: Scenario(phrase(quick=q), out_sr=v),
         lambda v: f"{v // 1000 if v % 1000 == 0 else v / 1000}k"),
    Axis('long_drift', 'two 20 s notes drifting flat by this much',
         [0, 30, 60, 100], _long_drift, lambda v: f"-{v}c"),
]}

PRIMARY = ['baseline', 'vibrato_depth', 'vibrato_rate', 'note_duration',
           'sixteenths_bpm', 'repeat_gap', 'harmony_db', 'snr_db', 'h1_db',
           'register', 'mp3_kbps']
SECONDARY = [a for a in AXES if a not in PRIMARY]


def _accompaniment(notes: Sequence[Note], duration: float, seed: int) -> np.ndarray:
    """Piano-like block chords (I-vi-IV-V) voiced around the melody's own
    register and re-struck every 0.5 s, so they compete for both pitch and
    onsets - the hardest kind of bleed."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * SR))
    out = np.zeros(n)
    centre = float(np.median([x.midi for x in notes]))
    prog = [(0, 4, 7), (9, 0, 4), (5, 9, 0), (7, 11, 2)]
    strike = int(0.5 * SR)
    fade = np.ones(strike)
    fade[-441:] = np.linspace(1.0, 0.0, 441)
    for k, start in enumerate(np.arange(0.0, duration, 0.5)):
        s = int(start * SR)
        e = min(n, s + strike)
        local = np.arange(e - s) / SR
        for pc in prog[(k // 2) % len(prog)]:
            m = pc + 12 * np.round((centre - pc) / 12.0)
            f = 440.0 * 2 ** ((m - 69) / 12.0)
            for h in range(1, 11):
                fh = f * h * np.sqrt(1 + 1e-4 * h * h)
                if fh > 0.45 * SR:
                    break
                out[s:e] += (0.6 ** (h - 1) * np.exp(-local * (1.5 + 0.8 * h))
                             * np.sin(2 * np.pi * fh * local
                                      + rng.uniform(0, 2 * np.pi))
                             * fade[:e - s])
    return out


def _delay(x: np.ndarray, seconds: float) -> np.ndarray:
    k = int(round(seconds * SR))
    return np.concatenate([np.zeros(k), x[:len(x) - k]]) if k else x


def _match_power(x: np.ndarray, ref_power: float, db: float) -> np.ndarray:
    return x * np.sqrt(ref_power / (np.mean(x ** 2) + 1e-20)) * 10 ** (db / 20)


def render_scenario(scn: Scenario, stem: Path, seed: int) -> Path:
    """Render a scenario to disk; returns the file the systems will read."""
    import soundfile as sf
    from scipy.signal import butter, fftconvolve, sosfiltfilt

    stem.parent.mkdir(parents=True, exist_ok=True)
    duration = max(n.offset for n in scn.notes) + TAIL_S
    lead = synth_voice(scn.notes, scn.voice, SR, seed, duration)
    p_lead = float(np.mean(lead ** 2))
    mix = lead.copy()

    if scn.unison_cents is not None:
        vp = replace(scn.voice, detune_cents=scn.voice.detune_cents + scn.unison_cents,
                     vibrato_hz=scn.voice.vibrato_hz * 1.07)
        double = _delay(synth_voice(scn.notes, vp, SR, seed + 1, duration), 0.02)
        mix += _match_power(double, p_lead, 0.0)
    if scn.harmony_db is not None and scn.alt_notes:
        vp = replace(scn.voice, vibrato_hz=scn.voice.vibrato_hz * 0.9)
        harm = _delay(synth_voice(scn.alt_notes, vp, SR, seed + 2, duration), 0.015)
        mix += _match_power(harm, p_lead, scn.harmony_db)
    if scn.snr_db is not None:
        mix += _match_power(_accompaniment(scn.notes, duration, seed + 3),
                            p_lead, -scn.snr_db)
    if scn.reverb_rt60:
        rng = np.random.default_rng(seed + 4)
        L = int(SR * scn.reverb_rt60 * 1.2)
        ir = rng.standard_normal(L) * np.exp(-6.91 * np.arange(L) / SR / scn.reverb_rt60)
        ir[:int(0.01 * SR)] = 0.0
        wet = fftconvolve(mix, ir)[:len(mix)]
        mix = mix + _match_power(wet, float(np.mean(mix ** 2)), -3.0)
    if scn.clip_drive_db is not None:
        mix = np.clip(mix / np.max(np.abs(mix)) * 10 ** (scn.clip_drive_db / 20),
                      -1.0, 1.0)
    if scn.band:
        sos = butter(4, [scn.band[0] / (SR / 2), scn.band[1] / (SR / 2)],
                     btype='band', output='sos')
        mix = sosfiltfilt(sos, mix)
    mix = mix / (np.max(np.abs(mix)) + 1e-12) * 10 ** (scn.peak_dbfs / 20)

    sr = SR
    if scn.out_sr != SR:
        import librosa
        mix = librosa.resample(mix, orig_sr=SR, target_sr=scn.out_sr)
        sr = scn.out_sr

    if scn.mp3_kbps:
        path = stem.with_suffix('.mp3')
        # libsndfile maps level linearly onto 320..32 kbps but rejects exactly
        # 1.0 ("Error set compression level"); 0.999 still lands on 32 kbps.
        level = float(np.clip((320 - scn.mp3_kbps) / 288.0, 0.0, 0.999))
        sf.write(str(path), mix, sr, format='MP3', subtype='MPEG_LAYER_III',
                 compression_level=level, bitrate_mode='CONSTANT')
        scn.info['kbps_measured'] = round(path.stat().st_size * 8 / duration / 1000, 1)
        decoded, _ = sf.read(str(path), dtype='float64')
        m = min(len(decoded), len(mix))
        a, b = np.abs(mix[:m]), np.abs(decoded[:m])
        lags = range(-2000, 2001, 10)
        scn.info['decoder_offset_ms'] = round(1000 * max(
            lags, key=lambda L: float(np.dot(a[max(0, -L):m - max(0, L)],
                                             b[max(0, L):m - max(0, -L)]))) / SR, 1)
    else:
        path = stem.with_suffix('.wav')
        sf.write(str(path), mix, sr, subtype='PCM_16')
    return path


# --------------------------------------------------------------------------
# Running the engine, and scoring
# --------------------------------------------------------------------------

ENGINE_SYSTEMS = ('ensemble', 'crepe', 'basic_pitch')


def run_engine(path: Path, voters: Optional[Sequence[str]] = None):
    """The real `PitchEngine.transcribe`, with each voter's output recorded
    on the way through, so the same inference can be re-decoded per voter."""
    from ..pitch.engine import EngineSettings, PitchEngine

    engine = PitchEngine(EngineSettings(voters=voters) if voters else EngineSettings())
    captured: Dict[str, Any] = {}
    for voter in engine.voters:
        def spy(audio, n_frames, _orig=voter.observe, _name=voter.name):
            out = _orig(audio, n_frames)
            captured[_name] = out
            captured['_audio'] = audio
            return out
        voter.observe = spy
    result = engine.transcribe(path)
    return engine, result, captured


def redecode(engine, outputs, audio):
    """Decode + segment a voter subset exactly as the engine would."""
    from ..pitch.fusion import decode
    n = outputs[0].n_frames
    frames = decode(outputs, engine.settings.fusion, None)
    notes = engine._segment(frames, engine._onset_matrix(outputs, n),
                            engine._attack_envelope(audio, n))
    return frames, notes


def _unpack(notes: Sequence[Note]):
    iv = np.array([[n.onset, n.offset] for n in notes], dtype=float).reshape(-1, 2)
    hz = 440.0 * 2 ** ((np.array([n.midi for n in notes], dtype=float) - 69) / 12)
    return iv, hz


def _is_octave(diff: np.ndarray) -> np.ndarray:
    diff = np.asarray(diff, dtype=float)
    return (np.abs(diff) > 0.5) & (np.abs(diff - 12 * np.round(diff / 12)) <= 0.5)


def note_scores(ref: Sequence[Note], est: Sequence[Note]) -> Dict[str, Any]:
    """Note F1 (onset 50 ms, pitch 50 c), F1 with offsets, octave errors,
    spurious short notes and fragmentation."""
    import mir_eval
    out: Dict[str, Any] = {'n_ref': len(ref), 'n_est': len(est), 'f1': 0.0,
                           'p': 0.0, 'r': 0.0, 'f1_off': 0.0, 'oct_note': 0.0,
                           'spurious': len(est) if not ref else 0,
                           'frag': 0.0, 'onset_mae_ms': None}
    if not ref or not est:
        return out
    ref_iv, ref_hz = _unpack(ref)
    est_iv, est_hz = _unpack(est)
    kw = dict(onset_tolerance=0.05, pitch_tolerance=50.0)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, est_iv, est_hz, offset_ratio=None, **kw)
    _, _, f_off, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, est_iv, est_hz, offset_ratio=0.2,
        offset_min_tolerance=0.05, **kw)
    matches = mir_eval.transcription.match_notes(
        ref_iv, ref_hz, est_iv, est_hz, offset_ratio=None, **kw)
    matched = {e for _, e in matches}

    overlap = np.maximum(0.0, np.minimum(ref_iv[:, 1:2], est_iv[None, :, 1])
                         - np.maximum(ref_iv[:, 0:1], est_iv[None, :, 0]))
    has = overlap.max(axis=0) > 0
    best = overlap.argmax(axis=0)
    diff = np.array([est[j].midi - ref[best[j]].midi for j in range(len(est))])
    octave = has & _is_octave(diff)
    dur = est_iv[:, 1] - est_iv[:, 0]
    unmatched = np.array([j not in matched for j in range(len(est))])
    per_ref = (overlap >= 0.03).sum(axis=1)
    errors = [est_iv[e, 0] - ref_iv[i, 0] for i, e in matches]
    out.update(p=float(p), r=float(r), f1=float(f), f1_off=float(f_off),
               oct_note=float(octave.sum() / max(1, has.sum())),
               spurious=int((unmatched & (dur < 0.1)).sum()),
               frag=float(per_ref[per_ref > 0].mean()) if (per_ref > 0).any() else 0.0,
               onset_mae_ms=round(1000 * float(np.mean(np.abs(errors))), 1)
               if errors else None)
    return out


def _ref_raster(ref: Sequence[Note], duration: float, n: int):
    _, freqs = notes_to_f0(ref, duration=duration)
    freqs = np.pad(freqs, (0, max(0, n - len(freqs))))[:n]
    return freqs > 0, hz_to_midi(freqs)


def frame_scores(ref, frames, duration) -> Dict[str, float]:
    n = len(frames.times)
    ref_v, ref_m = _ref_raster(ref, duration, n)
    est_v = np.asarray(frames.voiced, dtype=bool)
    both = ref_v & est_v
    diff = np.asarray(frames.midi)[both] - ref_m[both]
    return {'rpa': float(np.sum(np.abs(diff) <= 0.5) / max(1, ref_v.sum())),
            'oct_frame': float(_is_octave(diff).sum() / max(1, both.sum())),
            'vr': float(both.sum() / max(1, ref_v.sum())),
            'vfa': float((est_v & ~ref_v).sum() / max(1, (~ref_v).sum()))}


def voter_diagnostics(out, ref, duration) -> Dict[str, float]:
    """One voter on its own: argmax pitch and voicing (>0.5) vs the truth."""
    from ..pitch.grid import PITCHES
    n = out.n_frames
    ref_v, ref_m = _ref_raster(ref, duration, n)
    arg = PITCHES[np.argmax(out.salience, axis=1)]
    diff = arg[ref_v] - ref_m[ref_v]
    voicing = np.asarray(out.voicing)
    return {'rpa': round(float(np.mean(np.abs(diff) <= 0.5)) if diff.size else 0.0, 3),
            'oct': round(float(np.mean(_is_octave(diff))) if diff.size else 0.0, 3),
            'vr': round(float(np.mean(voicing[ref_v] > 0.5)) if ref_v.any() else 0.0, 3),
            'vfa': round(float(np.mean(voicing[~ref_v] > 0.5)) if (~ref_v).any() else 0.0, 3)}


def _to_notes(engine_notes) -> List[Note]:
    return [Note(onset=n.start, offset=n.end, midi=float(n.midi)) for n in engine_notes]


def _score(scn: Scenario, frames, engine_notes, duration) -> Dict[str, Any]:
    est = _to_notes(engine_notes)
    row = note_scores(scn.notes, est)
    if frames is not None:
        row.update(frame_scores(scn.notes, frames, duration))
    if scn.alt_notes:
        row['f1_alt'] = note_scores(scn.alt_notes, est)['f1']
    cents = sorted({round(float(getattr(n, 'pitch_cents', 0.0))) for n in engine_notes})
    row['cents_off_values'] = cents[:8]
    return row


def evaluate(path: Path, scn: Scenario, systems: Sequence[str],
             voters: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    duration = max(n.offset for n in scn.notes) + TAIL_S
    rows: List[Dict[str, Any]] = []
    if any(s in ENGINE_SYSTEMS for s in systems):
        t0 = time.perf_counter()
        engine, result, captured = run_engine(path, voters)
        elapsed = time.perf_counter() - t0
        outputs = [captured[v.name] for v in engine.voters if v.name in captured]
        diag = {o.name: voter_diagnostics(o, scn.notes, duration) for o in outputs}
        for system in systems:
            if system == 'ensemble':
                row = _score(scn, result.frames, result.notes, duration)
                row['voters'] = diag
                row['runtime_s'] = round(elapsed, 2)
                try:   # proves the per-voter re-decode path is faithful
                    _, again = redecode(engine, outputs, captured['_audio'])
                    row['redecode_consistent'] = (
                        [(n.midi, round(n.start, 3), round(n.end, 3)) for n in again]
                        == [(n.midi, round(n.start, 3), round(n.end, 3))
                            for n in result.notes])
                except Exception as exc:   # engine internals renamed upstream
                    row['redecode_consistent'] = f"unavailable: {exc}"
            elif system in ('crepe', 'basic_pitch'):
                subset = [o for o in outputs if o.name == system]
                if not subset:
                    continue
                frames, notes = redecode(engine, subset, captured['_audio'])
                row = _score(scn, frames, notes, duration)
            else:
                continue
            row['system'] = system
            rows.append(row)
    if 'bp_notes' in systems:
        from .systems import BasicPitchSystem
        pred = BasicPitchSystem().run(path)
        row = note_scores(scn.notes, pred.notes or [])
        if scn.alt_notes:
            row['f1_alt'] = note_scores(scn.alt_notes, pred.notes or [])['f1']
        row.update(system='bp_notes', runtime_s=round(pred.runtime_s, 2))
        rows.append(row)
    return rows


def run_point(axis: str, value, out_dir: Path,
              systems: Sequence[str] = ('ensemble',),
              quick: bool = False,
              voters: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Render and score one point of one axis. `voters` overrides the
    engine's auto voter set (tests use it to stay fast)."""
    spec = AXES[axis]
    label = spec.label(value)
    scn = spec.build(value, quick)
    safe = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in label)
    stem = Path(out_dir) / 'audio' / axis / f"{axis}_{safe}{'_quick' if quick else ''}"
    path = render_scenario(scn, stem, seed_for(axis, label))
    rows = evaluate(path, scn, systems, voters)
    for row in rows:
        row.update(axis=axis, label=label, value=value, audio=str(path),
                   info=dict(scn.info))
    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _f(x, nd=3):
    return '-' if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def breaking_points(rows: List[Dict], system: str = 'ensemble') -> Dict[str, Dict]:
    """Per axis, where F1 first drops below 0.9 and below 0.7: a list of
    (first failing point, last good one) - (None, last point) if it never
    drops - for each direction away from the axis's best point.

    Walked outward from the best-scoring point both ways: most axes run easy
    -> extreme, but register, detune and sample rate are two-sided - the easy
    setting sits inside the list, and they can fail in both directions. Read
    from the first point only, the far side's failure went unreported.
    """
    out: Dict[str, Dict] = {}
    for axis in dict.fromkeys(r['axis'] for r in rows):
        pts = [r for r in rows if r['axis'] == axis and r['system'] == system]
        if not pts:
            continue
        best = max(range(len(pts)), key=lambda k: pts[k]['f1'])

        def first_below(th):
            if pts[best]['f1'] < th:
                return [(pts[best]['label'], None)]
            found = []
            for side in (range(best + 1, len(pts)), range(best - 1, -1, -1)):
                ok = pts[best]['label']
                for k in side:
                    if pts[k]['f1'] < th:
                        found.append((pts[k]['label'], ok))
                        break
                    ok = pts[k]['label']
                else:
                    if len(side):
                        found.append((None, ok))
            return found or [(None, pts[best]['label'])]
        out[axis] = {'lt0.9': first_below(0.9), 'lt0.7': first_below(0.7),
                     'range': f"{pts[0]['label']} -> {pts[-1]['label']}"}
    return out


def sweep_markdown(rows: List[Dict]) -> str:
    lines = ['# meloScribe stress sweep', '',
             'Note F1: onset 50 ms, pitch 50 cents (mir_eval). `F1off` also '
             'requires offsets (20% / 50 ms). `oct` = share of estimated notes '
             'an octave off the reference they overlap; `spur` = unmatched '
             'estimated notes < 100 ms; `frag` = estimated notes per reference '
             'note. Voter columns: that voter alone, argmax pitch accuracy '
             '(RPA) / octave share / voicing recall, from the same inference.',
             '', '## Breaking points', '',
             '| axis | easy -> extreme | ens F1<0.9 at | ens F1<0.7 at | '
             'crepe-only <0.9 | bp-only <0.9 |', '|---|---|---|---|---|---|']
    bp = {s: breaking_points(rows, s) for s in ENGINE_SYSTEMS}

    def fmt(entry):
        return '; '.join(f"**{hit}** (ok at {last})" if hit else
                         f"never (ok to {last})" for hit, last in entry)
    for axis, e in bp['ensemble'].items():
        extra = [fmt(bp[s][axis]['lt0.9']) if axis in bp[s] else '-'
                 for s in ('crepe', 'basic_pitch')]
        lines.append(f"| {axis} | {e['range']} | {fmt(e['lt0.9'])} | "
                     f"{fmt(e['lt0.7'])} | {extra[0]} | {extra[1]} |")
    for axis in dict.fromkeys(r['axis'] for r in rows):
        desc = AXES[axis].description if axis in AXES else ''
        lines += ['', f"## {axis} - {desc}", '',
                  '| point | F1 | F1off | P | R | oct | octF | spur | frag | '
                  'est/ref | VFA | crepe F1 | bp F1 | crepe RPA/oct/VR | '
                  'bp RPA/oct/VR | extra |',
                  '|' + '---|' * 16]
        for r in [r for r in rows if r['axis'] == axis and r['system'] == 'ensemble']:
            sub = {s['system']: s for s in rows
                   if s['axis'] == axis and s['label'] == r['label']}
            v = r.get('voters', {})

            def vd(name):
                d = v.get(name)
                return f"{d['rpa']:.2f}/{d['oct']:.2f}/{d['vr']:.2f}" if d else '-'
            extra = []
            if 'f1_alt' in r:
                extra.append(f"F1 vs harmony {r['f1_alt']:.2f}")
            extra += [f"{k}={val}" for k, val in r.get('info', {}).items()]
            if r.get('onset_mae_ms') is not None:
                extra.append(f"onsetMAE {r['onset_mae_ms']}ms")
            lines.append(
                f"| {r['label']} | {r['f1']:.3f} | {r['f1_off']:.3f} | "
                f"{r['p']:.2f} | {r['r']:.2f} | {r['oct_note']:.2f} | "
                f"{_f(r.get('oct_frame'), 2)} | {r['spurious']} | {r['frag']:.2f} | "
                f"{r['n_est']}/{r['n_ref']} | {_f(r.get('vfa'), 2)} | "
                f"{_f(sub.get('crepe', {}).get('f1'))} | "
                f"{_f(sub.get('basic_pitch', {}).get('f1'))} | {vd('crepe')} | "
                f"{vd('basic_pitch')} | {'; '.join(extra)} |")
    return '\n'.join(lines) + '\n'


def cmd_sweep(args) -> int:
    out_dir = Path(args.out)
    axes = (PRIMARY + SECONDARY if args.axis == 'all'
            else SECONDARY if args.axis == 'secondary'
            else [a.strip() for a in args.axis.split(',')] if args.axis else PRIMARY)
    systems = [s.strip() for s in args.systems.split(',') if s.strip()]
    json_path = out_dir / f"{args.tag}.json"
    rows: List[Dict] = []
    for axis in axes:
        if axis not in AXES:
            print(f"unknown axis {axis!r}; see `list`", file=sys.stderr)
            return 2
        for value in AXES[axis].points:
            try:
                new = run_point(axis, value, out_dir, systems, quick=args.quick)
            except Exception as exc:
                traceback.print_exc()
                new = [{'axis': axis, 'label': AXES[axis].label(value),
                        'value': value, 'system': 'ensemble', 'f1': 0.0,
                        'f1_off': 0.0, 'p': 0.0, 'r': 0.0, 'oct_note': 0.0,
                        'spurious': 0, 'frag': 0.0, 'n_est': 0, 'n_ref': 0,
                        'error': f"{type(exc).__name__}: {exc}"}]
            rows.extend(new)
            ens = next((r for r in new if r['system'] == 'ensemble'), new[0])
            print(f"{axis:15s} {ens['label']:14s} F1={ens['f1']:.3f} "
                  f"F1off={ens['f1_off']:.3f} oct={ens['oct_note']:.2f} "
                  f"spur={ens['spurious']} frag={ens['frag']:.2f} "
                  f"est/ref={ens['n_est']}/{ens['n_ref']}  "
                  + ' '.join(f"{r['system']}={r['f1']:.2f}" for r in new
                             if r['system'] != 'ensemble'), flush=True)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(rows, indent=1, default=str))
    (out_dir / f"{args.tag}.md").write_text(sweep_markdown(rows), encoding='utf-8')
    print(f"\n{json_path}\n{out_dir / (args.tag + '.md')}")
    return 0


# --------------------------------------------------------------------------
# Fuzzing the real entry points
# --------------------------------------------------------------------------

FUZZ_PHRASE = [(57, 0.3, 0.75), (60, 0.85, 1.3), (64, 1.4, 1.85), (69, 1.95, 2.4)]


def _fuzz_voice(duration: float = 2.9):
    notes = [Note(on, off, float(m)) for m, on, off in FUZZ_PHRASE]
    x = synth_voice(notes, VoiceParams(), SR, seed_for('fuzz'), duration)
    return 0.8 * x / np.max(np.abs(x)), notes


def _write_mp3(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    import soundfile as sf
    sf.write(str(path), x, sr, format='MP3', subtype='MPEG_LAYER_III')
    return path


def make_fuzz_case(name: str, d: Path) -> Tuple[Path, str, List[Note]]:
    """Returns (file, expectation, reference notes). Expectations: 'none'
    (no notes), 'phrase', 'first_half', 'error' (undecodable: a clean
    error is the right answer)."""
    import soundfile as sf
    d.mkdir(parents=True, exist_ok=True)
    v, notes = _fuzz_voice()
    rng = np.random.default_rng(seed_for('fuzz', name))
    wav = lambda p, x, **kw: (sf.write(str(p), x, SR, **kw), p)[1]   # noqa: E731

    if name == 'silence':
        return wav(d / 'silence.wav', np.zeros(3 * SR), subtype='PCM_16'), 'none', []
    if name == 'hum_-70dBFS':
        t = np.arange(3 * SR) / SR
        hum = sum(np.sin(2 * np.pi * 60 * h * t) / h for h in (1, 2, 3, 4))
        hum = hum / np.max(np.abs(hum)) * 10 ** (-70 / 20)
        return wav(d / 'hum.wav', hum, subtype='PCM_16'), 'none', []
    if name == 'empty_wav':
        return wav(d / 'empty.wav', np.zeros(0), subtype='PCM_16'), 'error', []
    if name == 'tiny_2ms':
        return wav(d / 'tiny.wav', v[int(0.5 * SR):int(0.502 * SR)],
                   subtype='PCM_16'), 'any', []
    if name == 'short_0.2s':
        x = synth_voice([Note(0.02, 0.18, 69.0)], VoiceParams(), SR, 7, 0.2)
        x = 0.8 * x / np.max(np.abs(x))
        return (wav(d / 'short.wav', x, subtype='PCM_16'), 'phrase',
                [Note(0.02, 0.18, 69.0)])
    if name == 'mono':
        return wav(d / 'mono.wav', v, subtype='PCM_16'), 'phrase', notes
    if name == 'stereo':
        st = np.stack([v, 0.9 * v + 0.001 * rng.standard_normal(len(v))], axis=1)
        return wav(d / 'stereo.wav', st, subtype='PCM_16'), 'phrase', notes
    if name == 'phase_inverted_stereo':
        return (wav(d / 'phase_inverted.wav', np.stack([v, -v], axis=1),
                    subtype='PCM_16'), 'phrase', notes)
    if name == 'wav_u8':
        return wav(d / 'u8.wav', v, subtype='PCM_U8'), 'phrase', notes
    if name == 'wav_float32_overrange':
        return wav(d / 'float_over.wav', 2.0 * v, subtype='FLOAT'), 'phrase', notes
    if name in ('nan_sample', 'inf_sample'):
        x = v.copy()
        x[len(x) // 2] = np.nan if name == 'nan_sample' else np.inf
        return wav(d / f"{name}.wav", x, subtype='FLOAT'), 'phrase', notes
    if name == 'dc_offset':
        return wav(d / 'dc.wav', 0.6 * v + 0.35, subtype='PCM_16'), 'phrase', notes
    if name == 'mp3_id3_art':
        from mutagen.id3 import APIC, ID3, TIT2, TPE1
        p = _write_mp3(d / 'tagged.mp3', v)
        tags = ID3()
        tags.add(TIT2(encoding=3, text='Stress'))
        tags.add(TPE1(encoding=3, text='Nobody'))
        tags.add(APIC(encoding=3, mime='image/jpeg', type=3, desc='cover',
                      data=b'\xff\xd8\xff\xe0' + rng.bytes(200_000)))
        tags.save(str(p))
        return p, 'phrase', notes
    if name == 'mp3_truncated':
        p = _write_mp3(d / 'full.mp3', v)
        cut = d / 'truncated.mp3'
        data = p.read_bytes()
        cut.write_bytes(data[:len(data) // 2 + 7])      # mid-frame
        return cut, 'first_half', notes
    if name == 'mp3_garbage':
        p = d / 'garbage.mp3'
        p.write_bytes(rng.bytes(50_000))
        return p, 'error', []
    if name == 'wav_named_mp3':
        p = d / 'actually_wav.mp3'
        sf.write(str(p), v, SR, format='WAV', subtype='PCM_16')
        return p, 'phrase', notes
    if name == 'mp3_named_wav':
        p = d / 'actually_mp3.wav'
        sf.write(str(p), v, SR, format='MP3', subtype='MPEG_LAYER_III')
        return p, 'phrase', notes
    if name == 'unicode_name':
        return (wav(d / 'Café del Mar – ñandú 你好 (take 2).wav', v,
                    subtype='PCM_16'), 'phrase', notes)
    raise ValueError(f"unknown fuzz case {name!r}")


FUZZ_CASES = ['silence', 'hum_-70dBFS', 'empty_wav', 'tiny_2ms', 'short_0.2s',
              'mono', 'stereo', 'phase_inverted_stereo', 'wav_u8',
              'wav_float32_overrange', 'nan_sample', 'inf_sample', 'dc_offset',
              'mp3_id3_art', 'mp3_truncated', 'mp3_garbage', 'wav_named_mp3',
              'mp3_named_wav', 'unicode_name']
ENTRIES = ('engine', 'pipeline', 'cli')


def _entry(entry: str, path: Path, voters: Optional[Sequence[str]] = None
           ) -> Tuple[List[Tuple[int, float, float]], Dict]:
    if entry == 'engine':
        from ..pitch.engine import EngineSettings, PitchEngine
        res = PitchEngine(EngineSettings(voters=voters) if voters
                          else EngineSettings()).transcribe(path)
        return [(n.midi, n.start, n.end) for n in res.notes], {}
    if entry == 'pipeline':
        from ..pipeline import Pipeline, TranscriptionRequest
        out = Pipeline().run(TranscriptionRequest(
            input_path=path, vocals_only=True, lyrics_mode='off'))
        return ([(n.midi, n.start, n.end) for n in out.notes],
                {'warnings': out.warnings[:3]})
    if entry == 'cli':
        from .. import cli
        so, se = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = cli.main([str(path), '--vocals-only', '--no-lyrics', '--quiet',
                           '--format', 'json'])
        meta: Dict[str, Any] = {'rc': rc, 'stderr': se.getvalue().strip()[-300:]}
        text = so.getvalue()
        notes: List[Tuple[int, float, float]] = []
        if rc == 0:
            try:
                data = json.loads(text)
                meta['stdout_is_json'] = True
            except ValueError:
                meta['stdout_is_json'] = False
                meta['stdout_head'] = text[:120]
                data = json.loads(text[text.index('{'):]) if '{' in text else {}
            notes = [(n['midi'], n['start_time'], n['end_time'])
                     for n in data.get('notes', [])]
        return notes, meta
    raise ValueError(entry)


class _Timeout(Exception):
    pass


def _judge(expect: str, ref: List[Note], got, cut_s: Optional[float] = None) -> str:
    est = [Note(s, e, float(m)) for m, s, e in got]
    if expect == 'none':
        return 'ok' if not est else f"WRONG: {len(est)} notes in silence"
    if expect in ('error', 'any'):
        return 'ok (no notes)' if not est else f"ok ({len(est)} notes)"
    if expect == 'first_half' and cut_s is not None:
        ref = [n for n in ref if n.offset <= cut_s]
    if not est:
        return 'WRONG: no notes'
    s = note_scores(ref, est)
    if s['oct_note'] > 0:
        return f"WRONG: octave errors ({s['oct_note']:.2f})"
    if s['f1'] < 0.75:
        return f"WRONG: F1={s['f1']:.2f} ({len(est)} notes)"
    return f"ok F1={s['f1']:.2f}"


def run_fuzz_case(name: str, out_dir: Path, entries: Sequence[str] = ENTRIES,
                  timeout_s: int = 300,
                  voters: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    path, expect, ref = make_fuzz_case(name, Path(out_dir) / 'audio' / 'fuzz' / name)
    cut_s = None
    if expect == 'first_half':
        import soundfile as sf
        try:
            # What decodes, not what the header claims: a truncated MP3's
            # Xing header still gives the whole song's length (2.9 s of which
            # 1.39 s decodes), and judging against that marked a perfect
            # transcription of what is there as WRONG.
            samples, sr = sf.read(str(path))
            cut_s = len(samples) / sr
        except Exception:
            cut_s = 1.3
    rows = []
    for entry in entries:
        row: Dict[str, Any] = {'case': name, 'entry': entry, 'file': str(path),
                               'expect': expect}
        t0 = time.perf_counter()

        def on_alarm(signum, frame):
            raise _Timeout(f"no result after {timeout_s}s")
        # SIGALRM is POSIX-only. On Windows a case runs without the timeout,
        # so a hang stalls the run instead of being reported as HANG.
        alarm = hasattr(signal, 'SIGALRM')
        if alarm:
            previous = signal.signal(signal.SIGALRM, on_alarm)
            signal.alarm(timeout_s)
        try:
            got, meta = _entry(entry, path, voters)
            row.update(meta)
            row['notes'] = [f"{note_name(m)}@{s:.2f}-{e:.2f}" for m, s, e in got][:10]
            row['n_notes'] = len(got)
            if entry == 'cli' and meta.get('rc'):
                row['verdict'] = ('handled error' if expect == 'error'
                                  else f"FAIL: rc={meta['rc']}")
            else:
                row['verdict'] = _judge(expect, ref, got, cut_s)
                if entry == 'cli' and meta.get('stdout_is_json') is False:
                    row['verdict'] += ' | stdout not pure JSON'
        except _Timeout as exc:
            row.update(verdict='HANG', error=str(exc))
        except BaseException as exc:   # SystemExit included: the CLI must not die
            tb = traceback.extract_tb(exc.__traceback__)
            row.update(
                verdict=('raised (expected)' if expect == 'error' and entry != 'cli'
                         else 'CRASH'),
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
                where=[f"{Path(fr.filename).name}:{fr.lineno} {fr.name}"
                       for fr in tb[-4:]])
        finally:
            if alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
        row['seconds'] = round(time.perf_counter() - t0, 1)
        rows.append(row)
    return rows


def fuzz_markdown(rows: List[Dict]) -> str:
    lines = ['# meloScribe input fuzzing', '',
             '| case | entry | verdict | notes | error / where |',
             '|---|---|---|---|---|']
    for r in rows:
        err = ''
        if r.get('error'):
            err = f"`{r['error'][:140]}` at {' <- '.join(reversed(r.get('where', [])))}"
        elif r.get('stderr') and r.get('rc'):
            err = r['stderr'][-140:].replace('\n', ' ')
        lines.append(f"| {r['case']} | {r['entry']} | {r['verdict']} | "
                     f"{', '.join(r.get('notes', [])[:5])} | {err} |")
    return '\n'.join(lines) + '\n'


def cmd_fuzz(args) -> int:
    out_dir = Path(args.out)
    cases = [c.strip() for c in args.case.split(',')] if args.case else FUZZ_CASES
    entries = [e.strip() for e in args.entries.split(',')]
    rows: List[Dict] = []
    for case in cases:
        new = run_fuzz_case(case, out_dir, entries, args.timeout)
        rows.extend(new)
        for r in new:
            print(f"{case:24s} {r['entry']:9s} {r['verdict']:40s} "
                  f"{r.get('error', '')[:90]}", flush=True)
        (out_dir / f"{args.tag}.json").write_text(json.dumps(rows, indent=1, default=str))
    (out_dir / f"{args.tag}.md").write_text(fuzz_markdown(rows), encoding='utf-8')
    print(f"\n{out_dir / (args.tag + '.json')}\n{out_dir / (args.tag + '.md')}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    sw = sub.add_parser('sweep', help='capability-envelope sweeps')
    sw.add_argument('--axis', help="comma list, 'all' or 'secondary' "
                                   "(default: the primary axes)")
    sw.add_argument('--systems', default='ensemble,crepe,basic_pitch',
                    help='ensemble, crepe, basic_pitch (engine re-decodes, free) '
                         'and bp_notes (raw basic-pitch baseline)')
    sw.add_argument('--quick', action='store_true', help='short phrases')
    sw.add_argument('--out', default=str(DEFAULT_OUT))
    sw.add_argument('--tag', default='sweep')
    fz = sub.add_parser('fuzz', help='hostile inputs through the real entry points')
    fz.add_argument('--case', help='comma list (default: all)')
    fz.add_argument('--entries', default=','.join(ENTRIES))
    fz.add_argument('--timeout', type=int, default=300,
                    help='seconds before a case is reported as HANG '
                         '(POSIX only: ignored on Windows)')
    fz.add_argument('--out', default=str(DEFAULT_OUT))
    fz.add_argument('--tag', default='fuzz')
    sub.add_parser('list', help='list axes and fuzz cases')
    args = parser.parse_args(argv)

    if args.cmd == 'list':
        for name, a in AXES.items():
            tier = 'primary' if name in PRIMARY else 'secondary'
            print(f"{name:15s} {tier:9s} {a.description}: "
                  f"{', '.join(a.label(p) for p in a.points)}")
        print('\nfuzz cases:', ', '.join(FUZZ_CASES))
        return 0
    return cmd_sweep(args) if args.cmd == 'sweep' else cmd_fuzz(args)


if __name__ == '__main__':
    sys.exit(main())
