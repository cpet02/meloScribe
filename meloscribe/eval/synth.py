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
                  seed: int = 0) -> Tuple[np.ndarray, float]:
    """Render a note list to audio, optionally over a chordal backing.

    `backing_level` simulates imperfect stem separation: a pad playing triads
    underneath the melody, at the given linear amplitude relative to the voice.
    That bleed is what makes real stems harder than clean synthesis.
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


def build_case(case: Dict, out_dir: Path,
               sr: int = SAMPLE_RATE) -> GroundTruth:
    """Render one case to a WAV file and return its exact ground truth."""
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{case['name']}.wav"

    notes = build_notes(case['melody'])
    voice = PRESETS[case['voice']]
    # Seed from the case name so a given case is byte-identical run to run.
    seed = abs(hash(case['name'])) % (2 ** 31)

    audio, duration = render_melody(notes, voice, sr,
                                    backing_level=case.get('backing', 0.0),
                                    seed=seed)
    sf.write(str(wav_path), audio, sr)

    times, freqs = notes_to_f0(notes, duration=duration)
    return GroundTruth(
        name=case['name'],
        audio_path=wav_path,
        times=times,
        freqs=freqs,
        notes=notes,
        meta={'voice': case['voice'], 'melody': case['melody'],
              'backing': case.get('backing', 0.0), 'synthetic': True},
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
            notes = build_notes(case['melody'])
            import soundfile as sf
            info = sf.info(str(wav_path))
            times, freqs = notes_to_f0(notes, duration=info.duration)
            truths.append(GroundTruth(
                name=case['name'], audio_path=wav_path, times=times,
                freqs=freqs, notes=notes,
                meta={'voice': case['voice'], 'melody': case['melody'],
                      'backing': case.get('backing', 0.0), 'synthetic': True},
            ))
        else:
            truths.append(build_case(case, out_dir))

    return truths
