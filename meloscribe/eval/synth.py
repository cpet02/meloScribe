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
from dataclasses import dataclass
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
}


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

    if voice.breath_noise > 0:
        out += voice.breath_noise * rng.standard_normal(n)

    return out * _adsr(n, sr, voice.attack_s, voice.release_s)


def render_melody(notes: Sequence[Note], voice: Voice,
                  sr: int = SAMPLE_RATE,
                  backing_level: float = 0.0,
                  seed: int = 0,
                  pulse_bpm: float = 0.0,
                  pulse_level: float = 0.0) -> Tuple[np.ndarray, float]:
    """Render a note list to audio, optionally over a chordal backing.

    `backing_level` simulates imperfect stem separation: a pad playing triads
    underneath the melody, at the given linear amplitude relative to the voice.
    That bleed is what makes real stems harder than clean synthesis.

    `pulse_bpm`/`pulse_level` add a percussive pulse. Rhythm cases need it:
    beat tracking keys off percussive onsets, and a bare sung line gives a beat
    tracker nothing to lock onto, so a melody rendered without a pulse would
    test the plausibility score against a grid that was never really found.
    """
    rng = np.random.default_rng(seed)
    duration = max(n.offset for n in notes) + 0.5
    audio = np.zeros(int(duration * sr))

    for note in notes:
        rendered = render_note(note.midi, note.duration, voice, sr, rng)
        start = int(note.onset * sr)
        end = min(start + len(rendered), len(audio))
        audio[start:end] += rendered[:end - start]

    if backing_level > 0:
        audio += backing_level * _render_backing(notes, sr, len(audio), rng)

    if pulse_level > 0 and pulse_bpm > 0:
        audio += pulse_level * _render_pulse(pulse_bpm, sr, len(audio), rng)

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
}

# Melodies that must be rendered without rests, or they stop testing the thing
# they exist to test.
LEGATO_MELODIES = {'repeats', 'legato'}


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
]


def case_notes(case: Dict) -> List[Note]:
    """The ground-truth notes for a case, metrical or not."""
    if 'pattern' in case:
        return build_metrical_case(case)
    return build_notes(case['melody'])


def _case_meta(case: Dict) -> Dict:
    meta = {'voice': case['voice'], 'synthetic': True,
            'backing': case.get('backing', 0.0)}
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
                                    if 'pattern' in case else 0.0)
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
