"""Note segmentation: splitting held notes, and refusing to.

These are end-to-end through the pitch engine on synthesised audio, because the
bug they guard against only appears in the interaction between the onset
activation, the amplitude envelope and the singer's timbre - a unit test on the
envelope alone would have passed throughout.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from meloscribe.eval.synth import PRESETS, Voice, render_melody
from meloscribe.eval.groundtruth import Note
from meloscribe.pitch.engine import EngineSettings, PitchEngine


def render(tmp_path, notes, voice: Voice, name: str = 'case'):
    audio, _ = render_melody(notes, voice, seed=11)
    path = tmp_path / f"{name}.wav"
    sf.write(str(path), audio, 44100)
    return path


def transcribe(path, **kwargs):
    return PitchEngine(EngineSettings(**kwargs)).transcribe(path).notes


@pytest.fixture(scope='module')
def soft() -> Voice:
    """A softly-sung voice: slow attack, pronounced vibrato."""
    return PRESETS['soft']


def test_soft_sustained_note_is_not_shredded(tmp_path, soft):
    """The regression this exists for.

    The attack gate used to normalise the amplitude rise by the 99th percentile
    of rises across the track, which assumes the track contains a hard attack
    somewhere to set the scale. On softly-sung material it does not, the scale
    collapses to the size of the vibrato ripple, and a held note was split at
    the vibrato rate - one 1.0s note came out as five.
    """
    notes = [Note(onset=0.3, offset=2.3, midi=64.0)]
    produced = transcribe(render(tmp_path, notes, soft, 'sustained'))
    same_pitch = [n for n in produced if n.midi == 64]
    assert 1 <= len(same_pitch) <= 2, \
        f"one held note became {len(same_pitch)}: {[n.start for n in same_pitch]}"


def test_soft_rearticulation_is_still_split(tmp_path, soft):
    """The other side of the plateau: raising the gate until nothing splits
    would 'fix' the shredding by breaking re-articulation."""
    notes = [Note(onset=0.3, offset=0.9, midi=64.0),
             Note(onset=0.9, offset=1.5, midi=64.0),
             Note(onset=1.5, offset=2.1, midi=64.0)]
    produced = transcribe(render(tmp_path, notes, soft, 'repeats'))
    assert len([n for n in produced if n.midi == 64]) >= 3


@pytest.mark.parametrize('preset', ['clean', 'vocal', 'soft', 'scooped'])
def test_one_threshold_serves_every_attack_style(tmp_path, preset):
    """The same setting must work on a punched attack and a breathed one.

    This is the property the fix turns on, and the one the old envelope did not
    have. Normalising by a percentile of the track's own rises is already
    invariant to *gain* - what it is not invariant to is attack *shape*, so a
    threshold tuned on hard-attack material silently became far too sensitive
    on soft. Parametrising over timbres is what makes that visible; a single
    well-behaved voice passed either way.
    """
    notes = [Note(onset=0.3, offset=1.3, midi=64.0),
             Note(onset=1.5, offset=2.5, midi=67.0)]
    produced = transcribe(render(tmp_path, notes, PRESETS[preset], preset))
    assert len(produced) <= 3, \
        f"{preset}: two held notes became {len(produced)}"
    assert {n.midi for n in produced} >= {64, 67}
