"""A synthetic benchmark set with exact ground truth and no downloads.

Real annotated corpora (vocadito, MedleyDB) are the final word on accuracy, but
they are large, slow to score, and awkward to keep in a repo. This module
renders melodies whose ground truth we know to the sample, which buys us:

  - a regression check that runs in seconds on every change
  - failure modes we can dial in deliberately: octave-ambiguous timbres,
    vibrato, backing-track bleed, breathy noise
  - a floor test - a system that cannot score well here has a bug, not a
    tuning problem

Passing these is necessary, never sufficient. The real corpora still decide.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .groundtruth import GroundTruth, Note, midi_to_hz, notes_to_f0

SAMPLE_RATE = 44100


@dataclass
class Voice:
    """Timbre controls for the synthesised singer.

    `harmonic_rolloff` is the interesting knob: a low value gives a nearly pure
    tone with little harmonic structure, which is exactly when pitch trackers
    drop an octave. A high value gives a rich, easy-to-track voice.
    """
    n_harmonics: int = 12
    harmonic_rolloff: float = 0.7
    vibrato_hz: float = 5.0
    vibrato_cents: float = 25.0
    breath_noise: float = 0.02
    attack_s: float = 0.02
    release_s: float = 0.04
    even_harmonic_gain: float = 1.0
    # A scoop into the note: the pitch starts this many cents flat and glides
    # up to target over `scoop_ms`. Real singers do this constantly, and it is
    # the specific thing that defeats an onset taken from where the *pitch*
    # settled - the voice has been sounding for 80ms by then.
    scoop_cents: float = 0.0
    scoop_ms: float = 0.0
    # A creaky (vocal fry) onset: for the first `fry_ms` of each note, every
    # other glottal pulse is weaker, which puts a subharmonic at f0/2 under
    # the note. That period doubling is what creak sounds like acoustically,
    # and it is an octave trap aimed exactly at note onsets.
    fry_ms: float = 0.0
    fry_depth: float = 0.0
    # Portamento: notes that butt against each other are joined by a pitch
    # glide of this length, centred on the boundary, instead of being
    # re-attacked. Every legato case above re-articulates each note, so the
    # benchmark has never seen a voice that slides through the semitones in
    # between - each of which is a candidate spurious note.
    glide_ms: float = 0.0


PRESETS: Dict[str, Voice] = {
    # Rich and stable: everything should score near-perfectly.
    'clean': Voice(n_harmonics=14, harmonic_rolloff=0.75, vibrato_cents=10.0,
                   breath_noise=0.005),
    # Realistic pop vocal with expressive vibrato.
    'vocal': Voice(n_harmonics=10, harmonic_rolloff=0.65, vibrato_cents=35.0,
                   breath_noise=0.03),
    # Weak fundamental, mostly odd harmonics - the classic octave-error trap.
    'hollow': Voice(n_harmonics=6, harmonic_rolloff=0.35, vibrato_cents=15.0,
                    breath_noise=0.02, even_harmonic_gain=0.15),
    # Breathy and noisy: stresses voicing detection rather than pitch.
    'breathy': Voice(n_harmonics=8, harmonic_rolloff=0.5, vibrato_cents=40.0,
                     breath_noise=0.18),
    # A soft attack. Every voice above starts in 20ms, which makes an onset
    # trivial to place and is nothing like a sung entry.
    'soft': Voice(n_harmonics=10, harmonic_rolloff=0.65, vibrato_cents=30.0,
                  breath_noise=0.04, attack_s=0.11, release_s=0.09),
    # Soft attack *and* a scoop up to pitch. This is the combination that
    # exposes onset placement: the note is audible well before its pitch is
    # correct, so anything that waits for the pitch reports it late.
    'scooped': Voice(n_harmonics=10, harmonic_rolloff=0.65, vibrato_cents=30.0,
                     breath_noise=0.04, attack_s=0.09, release_s=0.09,
                     scoop_cents=140.0, scoop_ms=90.0),
    # Creaky entries: a period-doubled first 100ms on every note.
    'creaky': Voice(n_harmonics=10, harmonic_rolloff=0.65, vibrato_cents=25.0,
                    breath_noise=0.03, attack_s=0.03, release_s=0.06,
                    fry_ms=100.0, fry_depth=0.8),
    # A slow, expressive slide between legato notes.
    'gliding': Voice(n_harmonics=10, harmonic_rolloff=0.65, vibrato_cents=30.0,
                     breath_noise=0.03, attack_s=0.04, release_s=0.08,
                     glide_ms=180.0),
}


@dataclass
class Bleed:
    """What a real separated vocal stem carries besides the lead singer.

    Demucs has a single "vocals" stem, so every backing vocal lands in it next
    to the lead, typically mixed 6-12dB under it; drum transients are the other
    residue separation leaves behind. The core CASES model neither - their
    `backing` is a steady instrumental pad an octave down - and HANDOFF lists
    both as known gaps. None of it enters the annotation, which stays the lead
    line alone, so any frame spent on the harmony or a drum hit is scored as
    the error it would be on a real song.
    """
    # Backing vocal: the lead moved by this many C-major scale steps (+2 a
    # third above, -2 a third below, +4 a fifth above, -5 a sixth below).
    # 0 disables it.
    harmony_steps: int = 0
    harmony_db: float = -9.0
    # How late the backing line runs, in seconds. 0 is the unison-rhythm case;
    # a lag leaves the backing singer sounding alone in the lead's rests and on
    # the previous chord tone just after the lead moves.
    harmony_delay: float = 0.0
    # Drum bleed: a hi-hat on every eighth and a snare on 2 and 4. 0 disables.
    drums_bpm: float = 0.0
    hat_db: float = -12.0
    snare_db: float = -6.0


_C_MAJOR = (0, 2, 4, 5, 7, 9, 11)


def diatonic_shift(midi: float, steps: int) -> float:
    """Move a pitch by scale steps in C major.

    Backing vocals harmonise diatonically: a "third above" is major on some
    degrees and minor on others. A fixed semitone offset would put the
    harmony out of key on half the notes, which no arranger writes and which
    would make the harmony easier to reject than a real one.
    """
    m = int(round(midi))
    octave, pc = divmod(m, 12)
    # A chromatic note is harmonised from the scale tone below it, keeping
    # its inflection - the benchmark melodies are diatonic, so this is rare.
    base = max(p for p in _C_MAJOR if p <= pc)
    degree = octave * 7 + _C_MAJOR.index(base) + steps
    new_octave, new_degree = divmod(degree, 7)
    return float(new_octave * 12 + _C_MAJOR[new_degree] + (pc - base))


def harmony_notes(notes: Sequence[Note], bleed: Bleed) -> List[Note]:
    """The backing-vocal line a Bleed describes, as notes."""
    if not bleed.harmony_steps:
        return []
    return [Note(onset=n.onset + bleed.harmony_delay,
                 offset=n.offset + bleed.harmony_delay,
                 midi=diatonic_shift(n.midi, bleed.harmony_steps))
            for n in notes]


def case_seed(name: str) -> int:
    """A stable seed derived from a case name.

    Not `hash()`: Python randomises string hashing per process, so a case's
    seed changed on every run. That was invisible while the seed only affected
    rendering - the WAV is written once and reused - but it silently
    desynchronised any case whose *annotation* depends on the seed: the rubato
    melody was re-timed on each run while the audio it was scored against
    stayed as first rendered, and its score wandered by 0.18 with no code
    change to explain it.
    """
    return zlib.crc32(name.encode('utf-8'))


def _adsr(n: int, sr: int, attack_s: float, release_s: float) -> np.ndarray:
    """A simple attack/release envelope, to avoid clicks at note boundaries."""
    env = np.ones(n)
    a = min(int(attack_s * sr), n // 2)
    r = min(int(release_s * sr), n // 2)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r)
    return env


def render_note(midi: float, duration: float, voice: Voice,
                sr: int = SAMPLE_RATE,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Additive-synthesise one sung note."""
    rng = rng or np.random.default_rng(0)
    n = max(1, int(duration * sr))
    t = np.arange(n) / sr

    # Vibrato modulates the phase, not the frequency directly, so the pitch
    # excursion stays exactly +/- vibrato_cents.
    depth = voice.vibrato_cents / 1200.0
    vib = depth * np.sin(2 * np.pi * voice.vibrato_hz * t)

    # The scoop is a separate, one-way glide added to the vibrato: starting
    # flat and rising to target, decaying exponentially so it is over by
    # roughly `scoop_ms` rather than ending in a corner.
    if voice.scoop_cents > 0 and voice.scoop_ms > 0:
        tau = voice.scoop_ms / 3000.0
        vib = vib - (voice.scoop_cents / 1200.0) * np.exp(-t / tau)

    base_hz = float(midi_to_hz(midi))
    f0 = base_hz * (2.0 ** vib)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    out = np.zeros(n)
    for h in range(1, voice.n_harmonics + 1):
        if base_hz * h > sr / 2:
            break  # above Nyquist: aliasing, not a harmonic
        gain = voice.harmonic_rolloff ** (h - 1)
        if h % 2 == 0:
            gain *= voice.even_harmonic_gain
        out += gain * np.sin(h * phase + rng.uniform(0, 2 * np.pi))

    peak = np.max(np.abs(out))
    if peak > 0:
        out /= peak

    if voice.fry_ms > 0 and voice.fry_depth > 0:
        # Weakening every other cycle is an amplitude modulation at f0/2,
        # which is exactly the half-integer sidebands of period doubling.
        # Held for fry_ms, then released over 30ms into clean phonation.
        hold = min(int(voice.fry_ms / 1000.0 * sr), n)
        fade = min(int(0.03 * sr), n - hold)
        creak = np.zeros(n)
        creak[:hold] = 1.0
        if fade > 0:
            creak[hold:hold + fade] = np.linspace(1.0, 0.0, fade)
        out *= 1.0 - voice.fry_depth * creak * 0.5 * (1.0 + np.cos(phase / 2.0))

    if voice.breath_noise > 0:
        out += voice.breath_noise * rng.standard_normal(n)

    return out * _adsr(n, sr, voice.attack_s, voice.release_s)


def render_line(notes: Sequence[Note], voice: Voice,
                sr: int = SAMPLE_RATE,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """One breath of legato singing: contiguous notes joined by pitch glides.

    A single continuous phase and a single envelope, so nothing re-attacks at
    the note boundaries - only the pitch moves, along a raised-cosine glide of
    `voice.glide_ms` centred on each boundary. Centring it is what keeps the
    annotation honest: the pitch crosses the midpoint between the two notes
    exactly at the annotated boundary.
    """
    rng = rng or np.random.default_rng(0)
    first = notes[0].onset
    n = max(1, int((notes[-1].offset - first) * sr))
    t = np.arange(n) / sr

    midi = np.full(n, float(notes[0].midi))
    half = voice.glide_ms / 2000.0
    for prev, nxt in zip(notes, notes[1:]):
        boundary = nxt.onset - first
        midi[t >= boundary + half] = nxt.midi
        inside = (t >= boundary - half) & (t < boundary + half)
        x = (t[inside] - (boundary - half)) / max(2 * half, 1e-9)
        midi[inside] = prev.midi + (nxt.midi - prev.midi) * 0.5 * (1 - np.cos(np.pi * x))

    vib = (voice.vibrato_cents / 100.0) * np.sin(2 * np.pi * voice.vibrato_hz * t)
    f0 = midi_to_hz(midi + vib)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    out = np.zeros(n)
    for h in range(1, voice.n_harmonics + 1):
        if float(np.max(f0)) * h > sr / 2:
            break
        gain = voice.harmonic_rolloff ** (h - 1)
        if h % 2 == 0:
            gain *= voice.even_harmonic_gain
        out += gain * np.sin(h * phase + rng.uniform(0, 2 * np.pi))

    peak = np.max(np.abs(out))
    if peak > 0:
        out /= peak
    if voice.breath_noise > 0:
        out += voice.breath_noise * rng.standard_normal(n)
    return out * _adsr(n, sr, voice.attack_s, voice.release_s)


def _legato_groups(notes: Sequence[Note]) -> List[List[Note]]:
    """Split a note list into runs of notes that butt against each other."""
    groups: List[List[Note]] = []
    for note in sorted(notes, key=lambda n: n.onset):
        if groups and abs(note.onset - groups[-1][-1].offset) < 1e-6:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def render_melody(notes: Sequence[Note], voice: Voice,
                  sr: int = SAMPLE_RATE,
                  backing_level: float = 0.0,
                  seed: int = 0,
                  pulse_bpm: float = 0.0,
                  pulse_level: float = 0.0,
                  bleed: Optional[Bleed] = None,
                  note_db: Optional[Sequence[float]] = None
                  ) -> Tuple[np.ndarray, float]:
    """Render a note list to audio, optionally over a chordal backing.

    `backing_level` simulates imperfect stem separation: a pad playing triads
    underneath the melody, at the given linear amplitude relative to the voice.
    That bleed is what makes real stems harder than clean synthesis.

    `pulse_bpm`/`pulse_level` add a percussive pulse. Rhythm cases need it:
    beat tracking keys off percussive onsets, and a bare sung line gives a beat
    tracker nothing to lock onto, so a melody rendered without a pulse would
    test the plausibility score against a grid that was never really found.

    `bleed` adds what a real vocal stem carries besides the lead - a backing
    vocal and drum residue - at levels relative to the lead's peak.

    `note_db` sings each note at its own level (not for a gliding voice,
    whose legato run is one line). Every other case sings every note equally
    loud, so nothing could tell a soft lead note from a backing voice at the
    same level - exactly the cost a level-based voicing cue would carry, and
    one the benchmark could not see.
    """
    rng = np.random.default_rng(seed)
    duration = max(n.offset for n in notes) + 0.5
    audio = np.zeros(int(duration * sr))

    # A gliding voice renders each legato run as one continuous line; every
    # other voice renders note by note, exactly as before glides existed.
    units = (_legato_groups(notes) if voice.glide_ms > 0
             else [[note] for note in notes])
    for i, unit in enumerate(units):
        if len(unit) == 1:
            rendered = render_note(unit[0].midi, unit[0].duration, voice, sr, rng)
        else:
            rendered = render_line(unit, voice, sr, rng)
        if note_db is not None and voice.glide_ms <= 0:
            rendered = rendered * 10.0 ** (note_db[i] / 20.0)
        start = int(unit[0].onset * sr)
        end = min(start + len(rendered), len(audio))
        audio[start:end] += rendered[:end - start]

    if backing_level > 0:
        audio += backing_level * _render_backing(notes, sr, len(audio), rng)

    if pulse_level > 0 and pulse_bpm > 0:
        audio += pulse_level * _render_pulse(pulse_bpm, sr, len(audio), rng)

    if bleed is not None:
        audio += _render_bleed(notes, voice, bleed, sr, len(audio), seed)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = 0.89 * audio / peak

    return audio, duration


def _render_backing(notes: Sequence[Note], sr: int, n_samples: int,
                    rng: np.random.Generator) -> np.ndarray:
    """A sustained triad pad an octave below the melody's centre.

    Deliberately placed in a register that overlaps the melody's lower
    harmonics, since that is where separation bleed actually confuses a
    pitch tracker.
    """
    centre = float(np.median([n.midi for n in notes]))
    root = centre - 12.0
    pad = np.zeros(n_samples)
    t = np.arange(n_samples) / sr

    for interval in (0, 4, 7):
        f = float(midi_to_hz(root + interval))
        for h in (1, 2, 3):
            pad += (0.4 ** h) * np.sin(2 * np.pi * f * h * t + rng.uniform(0, 6.28))

    peak = np.max(np.abs(pad))
    return pad / peak if peak > 0 else pad


def _render_pulse(bpm: float, sr: int, n_samples: int,
                  rng: np.random.Generator) -> np.ndarray:
    """A dry percussive click on every beat, with a stronger downbeat.

    Deliberately not a pure tone: a beat tracker responds to broadband
    transients, and the downbeat accent gives it a phase to lock to rather than
    just a period. Alternate beats are *not* dropped - that pattern is what
    makes a tracker report half tempo, which the diagnostics exist to catch and
    which the benchmark should therefore not build in by default.
    """
    pulse = np.zeros(n_samples)
    period = 60.0 / bpm
    click_n = int(0.03 * sr)
    envelope = np.exp(-np.arange(click_n) / (0.006 * sr))
    noise = rng.standard_normal(click_n) * envelope
    body = np.sin(2 * np.pi * 180.0 * np.arange(click_n) / sr) * envelope

    beat = 0
    while beat * period * sr < n_samples:
        start = int(beat * period * sr)
        end = min(start + click_n, n_samples)
        gain = 1.0 if beat % 4 == 0 else 0.6
        pulse[start:end] += gain * (0.6 * noise + 0.4 * body)[:end - start]
        beat += 1

    peak = np.max(np.abs(pulse))
    return pulse / peak if peak > 0 else pulse


def _render_bleed(notes: Sequence[Note], voice: Voice, bleed: Bleed, sr: int,
                  n_samples: int, seed: int) -> np.ndarray:
    """The backing vocal and drum residue a Bleed describes, lead excluded.

    Drawn from its own generator, so adding bleed to a case can never change
    how that case's lead line was rendered.
    """
    rng = np.random.default_rng([seed, 0xB1EED])
    out = np.zeros(n_samples)

    harmony = harmony_notes(notes, bleed)
    if harmony:
        # A second singer, not a transposed copy: vibrato at a different rate,
        # so the two lines drift in and out of phase as two real voices do.
        singer = replace(voice, vibrato_hz=voice.vibrato_hz * 1.17)
        gain = 10.0 ** (bleed.harmony_db / 20.0)
        for note in harmony:
            rendered = render_note(note.midi, note.duration, singer, sr, rng)
            start = int(note.onset * sr)
            end = min(start + len(rendered), n_samples)
            if end > start:
                out[start:end] += gain * rendered[:end - start]

    if bleed.drums_bpm > 0:
        out += _render_drums(bleed, sr, n_samples, rng)
    return out


def _render_drums(bleed: Bleed, sr: int, n_samples: int,
                  rng: np.random.Generator) -> np.ndarray:
    """A hi-hat on every eighth and a snare on 2 and 4, as stem residue.

    The hat is short high-passed noise: broadband, unpitched, a pure test of
    whether a transient fakes an attack. The snare adds two inharmonic body
    modes (a circular membrane's 1 : 1.59) at 190Hz - squarely inside the sung
    range, which is what makes snare bleed a pitch problem and not only a
    voicing one.
    """
    from scipy.signal import butter, sosfilt

    out = np.zeros(n_samples)
    eighth_s = 30.0 / bleed.drums_bpm
    hat_gain = 10.0 ** (bleed.hat_db / 20.0)
    snare_gain = 10.0 ** (bleed.snare_db / 20.0)

    hat_t = np.arange(int(0.05 * sr)) / sr
    hat_sos = butter(4, 3000.0 / (sr / 2), btype='high', output='sos')
    snare_t = np.arange(int(0.2 * sr)) / sr
    wire_sos = butter(2, [800.0 / (sr / 2), 8000.0 / (sr / 2)], btype='band',
                      output='sos')

    def hit(signal: np.ndarray, start: int, gain: float) -> None:
        signal = signal / (np.max(np.abs(signal)) + 1e-12)
        end = min(start + len(signal), n_samples)
        out[start:end] += gain * signal[:end - start]

    eighth = 0
    while eighth * eighth_s * sr < n_samples:
        start = int(eighth * eighth_s * sr)
        hat = sosfilt(hat_sos, rng.standard_normal(hat_t.size))
        hit(hat * np.exp(-hat_t / 0.012), start, hat_gain)
        if eighth % 4 == 2:  # beats 2 and 4 of each bar
            wires = sosfilt(wire_sos, rng.standard_normal(snare_t.size))
            wires *= np.exp(-snare_t / 0.07)
            body = (np.sin(2 * np.pi * 190.0 * snare_t)
                    + 0.5 * np.sin(2 * np.pi * 302.0 * snare_t))
            body *= np.exp(-snare_t / 0.05)
            hit(wires / (np.max(np.abs(wires)) + 1e-12)
                + 0.8 * body / (np.max(np.abs(body)) + 1e-12), start, snare_gain)
        eighth += 1
    return out


# Melodic material. Intervals are chosen to include the cases that break
# naive trackers: octave leaps, chromatic neighbours, and wide ranges.
_MELODIES: Dict[str, List[Tuple[int, float]]] = {
    'scale': [(60, 0.4), (62, 0.4), (64, 0.4), (65, 0.4),
              (67, 0.4), (69, 0.4), (71, 0.4), (72, 0.8)],
    'octaves': [(60, 0.5), (72, 0.5), (62, 0.5), (74, 0.5),
                (64, 0.5), (76, 0.5), (60, 1.0)],
    'chromatic': [(67, 0.3), (68, 0.3), (69, 0.3), (68, 0.3),
                  (67, 0.3), (66, 0.3), (67, 0.6)],
    'wide': [(48, 0.6), (72, 0.4), (55, 0.4), (79, 0.4),
             (50, 0.6), (67, 0.8)],
    'phrase': [(64, 0.5), (64, 0.25), (67, 0.75), (69, 0.5),
               (67, 0.5), (64, 0.5), (62, 1.0), (60, 1.0)],
    # Repeated notes with no rest between them. This is the only case that
    # tests re-articulation: a pitch tracker sees one long note where there are
    # three, so the split can only come from onset detection. Rendered with
    # gap=0 (see LEGATO_MELODIES), which is why it is listed separately.
    'repeats': [(67, 0.4), (67, 0.4), (67, 0.4), (65, 0.8),
                (64, 0.4), (64, 0.4), (62, 0.8)],
    'legato': [(60, 0.5), (62, 0.5), (62, 0.5), (64, 0.5),
               (64, 0.5), (64, 0.5), (65, 1.0)],
    # Steps and thirds with no repeated pitch, for the portamento case: a
    # glide between two equal pitches is no glide, and would silently merge
    # the pair into one note that the annotation says is two.
    'slides': [(60, 0.6), (64, 0.6), (62, 0.6), (67, 0.6),
               (65, 0.6), (64, 0.6), (60, 1.0)],
}

# Melodies that must be rendered without rests, or they stop testing the thing
# they exist to test.
LEGATO_MELODIES = {'repeats', 'legato', 'slides'}


def build_notes(melody: str, gap: Optional[float] = None,
                start: float = 0.25) -> List[Note]:
    """Turn a named melody into timed notes.

    Rests are inserted between notes unless the melody is a legato one, where
    notes must butt directly against each other - inserting a gap there would
    silently convert the re-articulation test into an ordinary one.
    """
    if melody not in _MELODIES:
        raise ValueError(f"Unknown melody {melody!r}. "
                         f"Available: {sorted(_MELODIES)}")
    if gap is None:
        gap = 0.0 if melody in LEGATO_MELODIES else 0.08
    notes: List[Note] = []
    t = start
    for midi, dur in _MELODIES[melody]:
        notes.append(Note(onset=t, offset=t + dur, midi=float(midi)))
        t += dur + gap
    return notes


# --------------------------------------------------------------------------
# Metrical material, for the rhythm work
# --------------------------------------------------------------------------
#
# Every melody above has arbitrary durations and no tempo whatsoever, so the
# benchmark could not express the question "is this rhythm plausible?" at all -
# it would have scored a correct transcription and a scrambled one identically.
# These melodies are written in beats against a stated BPM instead, with a
# percussive pulse rendered underneath so the beat tracker has something real
# to find.
#
# Durations are in *beats*. Rests are written as a pitch of None, because a
# metrical melody needs its rests to sit on the grid too - inserting an
# arbitrary inter-note gap the way `build_notes` does would push every
# subsequent onset off the beat and make correct material look implausible.

_METRICAL: Dict[str, List[Tuple[Optional[int], float]]] = {
    # Plain quarters and eighths: the floor case, should score near 1.
    'straight': [(60, 1.0), (62, 1.0), (64, 0.5), (65, 0.5), (67, 1.0),
                 (None, 1.0), (67, 0.5), (65, 0.5), (64, 1.0), (62, 1.0),
                 (60, 2.0), (None, 1.0), (64, 0.5), (64, 0.5), (67, 1.0),
                 (69, 2.0)],
    # Syncopation and dotted values - correct music that a naive "is it on a
    # quarter note" test would wrongly punish.
    'syncopated': [(67, 0.75), (69, 0.25), (67, 0.5), (64, 0.5), (62, 1.0),
                   (None, 0.5), (64, 0.75), (65, 0.25), (67, 1.5), (65, 0.5),
                   (64, 1.0), (62, 0.5), (60, 1.5), (None, 0.5), (60, 1.0),
                   (64, 1.0), (67, 2.0)],
    # Triplets: the reason the ratio set is not just powers of two.
    'triplets': [(60, 1 / 3), (62, 1 / 3), (64, 1 / 3), (65, 1.0),
                 (67, 1 / 3), (65, 1 / 3), (64, 1 / 3), (62, 1.0),
                 (64, 1 / 3), (65, 1 / 3), (67, 1 / 3), (69, 1.0),
                 (67, 2.0), (None, 1.0), (60, 2.0)],
    # Sixteenth-note runs against held notes: tests the short end of the
    # duration range, where everything is close to something.
    'busy': [(60, 0.25), (62, 0.25), (64, 0.25), (65, 0.25), (67, 1.0),
             (69, 0.25), (67, 0.25), (65, 0.25), (64, 0.25), (62, 1.0),
             (64, 0.25), (65, 0.25), (67, 0.25), (69, 0.25), (72, 2.0),
             (None, 1.0), (67, 1.0), (64, 1.0)],
}


def build_metrical_notes(pattern: str, bpm: float = 100.0,
                         start_beat: float = 0.0,
                         swing: float = 0.0,
                         rubato_beats: float = 0.0,
                         legato: float = 0.92,
                         seed: int = 0) -> List[Note]:
    """Lay a metrical pattern out in seconds at a given tempo.

    `swing` delays every off-beat eighth by that fraction of an eighth (0.33 is
    roughly triplet swing). `rubato_beats` adds independent Gaussian timing
    noise to each onset. Both exist to check that the plausibility score
    tolerates real expressive timing rather than only tolerating a sequencer -
    a metric that scores swung or rubato playing as implausible would fire on
    exactly the music people care most about getting right.

    `legato` shortens each note slightly so consecutive notes do not butt
    together, matching how the non-metrical melodies are rendered.
    """
    if pattern not in _METRICAL:
        raise ValueError(f"Unknown metrical pattern {pattern!r}. "
                         f"Available: {sorted(_METRICAL)}")

    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    notes: List[Note] = []
    beat = start_beat

    for midi, length in _METRICAL[pattern]:
        if midi is not None:
            position = beat
            # Swing displaces the second eighth of each beat, and only that -
            # applying it everywhere would just be a tempo change.
            if swing and abs((beat % 1.0) - 0.5) < 1e-6:
                position += swing * 0.5
            if rubato_beats:
                position += float(rng.normal(0.0, rubato_beats))
            onset = position * period
            notes.append(Note(onset=onset,
                              offset=onset + length * period * legato,
                              midi=float(midi)))
        beat += length

    # A leading count-in of silence would leave the first note at t=0, which
    # both the pitch engine and the beat tracker find awkward.
    shift = period * 2.0
    return [Note(onset=n.onset + shift, offset=n.offset + shift, midi=n.midi)
            for n in notes]


# Rhythm cases. Each states its true BPM so the tempo estimate can be graded,
# not merely used.
METRICAL_CASES: List[Dict] = [
    {'name': 'metrical_straight',  'pattern': 'straight',   'bpm': 100.0,
     'voice': 'clean'},
    {'name': 'metrical_syncopated', 'pattern': 'syncopated', 'bpm': 92.0,
     'voice': 'vocal'},
    {'name': 'metrical_triplets',  'pattern': 'triplets',   'bpm': 120.0,
     'voice': 'vocal'},
    {'name': 'metrical_busy',      'pattern': 'busy',       'bpm': 84.0,
     'voice': 'clean'},
    # Expressive timing: these must still score well, or the metric is
    # measuring "sounds like a sequencer" rather than "sounds like music".
    {'name': 'metrical_swing',     'pattern': 'straight',   'bpm': 110.0,
     'voice': 'vocal', 'swing': 0.33},
    {'name': 'metrical_rubato',    'pattern': 'syncopated', 'bpm': 96.0,
     'voice': 'vocal', 'rubato': 0.035},
    # A tempo trap: the melody is at 132 but the only percussion is a half-time
    # backbeat at 66, which is exactly what makes a beat tracker report the
    # wrong metrical level. Without a case like this the tempo diagnostics
    # could only ever be confirmed, never caught out - every other case here
    # has an unambiguous pulse, so a confidence measure that returned 1.0
    # unconditionally would have looked perfect.
    {'name': 'metrical_halftime',  'pattern': 'straight',   'bpm': 132.0,
     'voice': 'vocal', 'pulse_bpm': 66.0},
]


def build_metrical_case(case: Dict) -> List[Note]:
    return build_metrical_notes(case['pattern'], bpm=case['bpm'],
                                swing=case.get('swing', 0.0),
                                rubato_beats=case.get('rubato', 0.0),
                                seed=case_seed(case['name']))


# The benchmark set itself: melody x voice x bleed, chosen to cover each
# failure mode once rather than to be exhaustive.
CASES: List[Dict] = [
    {'name': 'scale_clean',       'melody': 'scale',     'voice': 'clean',   'backing': 0.0},
    {'name': 'phrase_vocal',      'melody': 'phrase',    'voice': 'vocal',   'backing': 0.0},
    {'name': 'octaves_hollow',    'melody': 'octaves',   'voice': 'hollow',  'backing': 0.0},
    {'name': 'chromatic_vocal',   'melody': 'chromatic', 'voice': 'vocal',   'backing': 0.0},
    {'name': 'wide_hollow',       'melody': 'wide',      'voice': 'hollow',  'backing': 0.0},
    {'name': 'phrase_breathy',    'melody': 'phrase',    'voice': 'breathy', 'backing': 0.0},
    {'name': 'phrase_bleed',      'melody': 'phrase',    'voice': 'vocal',   'backing': 0.35},
    {'name': 'octaves_bleed',     'melody': 'octaves',   'voice': 'vocal',   'backing': 0.35},
    # Re-articulation: only these two can tell whether onset-based splitting
    # is earning its place.
    {'name': 'repeats_vocal',     'melody': 'repeats',   'voice': 'vocal',   'backing': 0.0},
    {'name': 'legato_clean',      'melody': 'legato',    'voice': 'clean',   'backing': 0.0},
    # Onset placement. Every case above starts its notes in 20ms at the right
    # pitch, so onsets are trivially placeable and note F1 at a 25ms tolerance
    # was identical to F1 at 50ms - the benchmark could not express the
    # question at all. These two can.
    {'name': 'phrase_soft',       'melody': 'phrase',    'voice': 'soft',    'backing': 0.0},
    {'name': 'phrase_scooped',    'melody': 'phrase',    'voice': 'scooped', 'backing': 0.0},
    {'name': 'scale_scooped',     'melody': 'scale',     'voice': 'scooped', 'backing': 0.0},
]


# Stress cases: what a real separated vocal stem does that the core set above
# cannot express - HANDOFF's "known gaps". Kept out of CASES on purpose, so the
# core numbers stay comparable with every earlier run; select them with
# `runner --suite hard`. The annotation is always the lead line alone.
HARD_CASES: List[Dict] = [
    # Backing vocals in unison rhythm: a parallel harmony that starts and stops
    # with the lead, so timing cannot tell the two lines apart - only level.
    {'name': 'harmony_3above',      'melody': 'phrase', 'voice': 'vocal',
     'bleed': Bleed(harmony_steps=2, harmony_db=-6.0)},
    {'name': 'harmony_3below',      'melody': 'phrase', 'voice': 'vocal',
     'bleed': Bleed(harmony_steps=-2, harmony_db=-6.0)},
    {'name': 'harmony_6below',      'melody': 'scale',  'voice': 'clean',
     'bleed': Bleed(harmony_steps=-5, harmony_db=-9.0)},
    # Offset rhythm: the backing singer runs late, so it is heard alone in the
    # lead's rests and on the old chord tone just after the lead moves on.
    {'name': 'harmony_5above_late', 'melody': 'scale',  'voice': 'vocal',
     'bleed': Bleed(harmony_steps=4, harmony_db=-9.0, harmony_delay=0.15)},
    {'name': 'harmony_3above_late', 'melody': 'phrase', 'voice': 'soft',
     'bleed': Bleed(harmony_steps=2, harmony_db=-12.0, harmony_delay=0.2)},
    {'name': 'repeats_harmony',     'melody': 'repeats', 'voice': 'vocal',
     'bleed': Bleed(harmony_steps=-2, harmony_db=-9.0, harmony_delay=0.1)},
    # Drum residue: the only thing HPSS denoising removes, so the only cases
    # that can say whether it should be on.
    {'name': 'phrase_drums',        'melody': 'phrase',  'voice': 'vocal',
     'bleed': Bleed(drums_bpm=112.0)},
    {'name': 'repeats_drums',       'melody': 'repeats', 'voice': 'vocal',
     'bleed': Bleed(drums_bpm=96.0)},
    {'name': 'scale_drums_loud',    'melody': 'scale',   'voice': 'soft',
     'bleed': Bleed(drums_bpm=128.0, hat_db=-6.0, snare_db=0.0)},
    # Expressive onsets and transitions.
    {'name': 'slides_portamento',   'melody': 'slides',  'voice': 'gliding'},
    {'name': 'phrase_creaky',       'melody': 'phrase',  'voice': 'creaky'},
    # Dynamics: lead notes sung up to 12dB under their neighbours, the same
    # range the backing vocals above sit in. The counterweight to the harmony
    # cases - without it, "quiet means not the lead" would look free. The
    # legato one also re-articulates a repeated pitch from loud to soft.
    {'name': 'phrase_dynamics',     'melody': 'phrase',  'voice': 'vocal',
     'note_db': [0.0, -4.0, -10.0, 0.0, -6.0, -12.0, -3.0, -9.0]},
    {'name': 'legato_dynamics',     'melody': 'legato',  'voice': 'soft',
     'note_db': [0.0, -3.0, -10.0, 0.0, -8.0, -2.0, -12.0]},
]

# Named case sets for `runner --suite`. 'core' is the comparable headline set.
SUITES: Dict[str, List[Dict]] = {
    'core': CASES,
    'hard': HARD_CASES,
    'all': CASES + HARD_CASES,
}


def case_notes(case: Dict) -> List[Note]:
    """The ground-truth notes for a case, metrical or not."""
    if 'pattern' in case:
        return build_metrical_case(case)
    return build_notes(case['melody'])


def _case_meta(case: Dict) -> Dict:
    meta = {'voice': case['voice'], 'synthetic': True,
            'backing': case.get('backing', 0.0)}
    if case.get('bleed') is not None:
        bleed = case['bleed']
        meta['bleed'] = dict(bleed.__dict__)
        # Kept alongside the truth so a diagnostic can ask "was that frame on
        # the backing line?" rather than only "was it wrong?".
        meta['harmony_notes'] = harmony_notes(case_notes(case), bleed)
    if case.get('note_db') is not None:
        meta['note_db'] = list(case['note_db'])
    if 'pattern' in case:
        meta.update({'melody': case['pattern'], 'metrical': True,
                     'bpm': case['bpm'], 'swing': case.get('swing', 0.0),
                     'rubato': case.get('rubato', 0.0)})
    else:
        meta['melody'] = case['melody']
    return meta


def build_case(case: Dict, out_dir: Path,
               sr: int = SAMPLE_RATE) -> GroundTruth:
    """Render one case to a WAV file and return its exact ground truth."""
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{case['name']}.wav"

    notes = case_notes(case)
    voice = PRESETS[case['voice']]
    # Seed from the case name so a given case is byte-identical run to run.
    seed = case_seed(case['name'])

    audio, duration = render_melody(notes, voice, sr,
                                    backing_level=case.get('backing', 0.0),
                                    seed=seed,
                                    pulse_bpm=case.get('pulse_bpm',
                                                       case.get('bpm', 0.0)),
                                    pulse_level=case.get('pulse', 0.35)
                                    if 'pattern' in case else 0.0,
                                    bleed=case.get('bleed'),
                                    note_db=case.get('note_db'))
    sf.write(str(wav_path), audio, sr)

    times, freqs = notes_to_f0(notes, duration=duration)
    return GroundTruth(
        name=case['name'],
        audio_path=wav_path,
        times=times,
        freqs=freqs,
        notes=notes,
        meta=_case_meta(case),
    )


def build_dataset(out_dir, cases: Optional[List[Dict]] = None,
                  force: bool = False) -> List[GroundTruth]:
    """Render the whole synthetic set, reusing WAVs that already exist."""
    out_dir = Path(out_dir)
    cases = cases or CASES
    truths: List[GroundTruth] = []

    for case in cases:
        wav_path = out_dir / f"{case['name']}.wav"
        if wav_path.exists() and not force:
            # Regenerate the annotation (cheap) but keep the audio (not).
            notes = case_notes(case)
            import soundfile as sf
            info = sf.info(str(wav_path))
            times, freqs = notes_to_f0(notes, duration=info.duration)
            truths.append(GroundTruth(
                name=case['name'], audio_path=wav_path, times=times,
                freqs=freqs, notes=notes, meta=_case_meta(case),
            ))
        else:
            truths.append(build_case(case, out_dir))

    return truths
