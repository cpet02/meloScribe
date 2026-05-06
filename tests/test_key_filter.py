"""
Tests for Step 1: Key-aware note filtering.
All tests use synthetic data — no audio files required.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.note_mapper import NoteMapper, map_notes, detect_key


class TestFilterByKey(unittest.TestCase):

    def setUp(self):
        # Use threshold 0.0 so confidence never interferes with key-filter tests
        self.mapper = NoteMapper(confidence_threshold=0.0, apply_key_filter=True)
        self.c_major_key = {'key': 'C major', 'score': 0.9, 'candidates': [('C major', 0.9)]}
        self.f_sharp_minor_key = {'key': 'F# minor', 'score': 0.85, 'candidates': [('F# minor', 0.85)]}

    def _make_event(self, note, t=0.0):
        return {'note': note, 'start_time': t, 'end_time': t + 0.1, 'confidence': 0.9}

    # --- core behaviour ---

    def test_in_key_notes_kept(self):
        events = [self._make_event(n) for n in ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4']]
        result = self.mapper.filter_by_key(events, self.c_major_key)
        self.assertEqual(len(result), 7)

    def test_out_of_key_notes_removed(self):
        events = [
            self._make_event('C4'),   # in C major
            self._make_event('C#4'),  # not in C major
            self._make_event('G4'),   # in C major
            self._make_event('A#4'),  # not in C major
        ]
        result = self.mapper.filter_by_key(events, self.c_major_key)
        notes = [e['note'] for e in result]
        self.assertIn('C4', notes)
        self.assertIn('G4', notes)
        self.assertNotIn('C#4', notes)
        self.assertNotIn('A#4', notes)

    def test_high_octave_outliers_removed(self):
        """The spurious overtone hits (e.g. G7, F6) are out-of-key and get dropped.
        F# minor scale: F# G# A B C# D E  — so G and F are out of key."""
        events = [
            self._make_event('F#3'),
            self._make_event('G7'),   # overtone artifact — G not in F# minor
            self._make_event('F#3'),
        ]
        result = self.mapper.filter_by_key(events, self.f_sharp_minor_key)
        self.assertFalse(any(e['note'] == 'G7' for e in result))
        self.assertEqual(len(result), 2)

    def test_octave_does_not_matter_only_pitch_class(self):
        """C4 and C6 are both C — both kept in C major regardless of octave."""
        events = [self._make_event('C4'), self._make_event('C6'), self._make_event('C#5')]
        result = self.mapper.filter_by_key(events, self.c_major_key)
        notes = [e['note'] for e in result]
        self.assertIn('C4', notes)
        self.assertIn('C6', notes)
        self.assertNotIn('C#5', notes)

    # --- safety / edge cases ---

    def test_low_key_score_skips_filter(self):
        """When key detection confidence is below 0.6, nothing is removed."""
        weak_key = {'key': 'C major', 'score': 0.55, 'candidates': [('C major', 0.55)]}
        events = [self._make_event('C#4'), self._make_event('G#4')]
        result = self.mapper.filter_by_key(events, weak_key)
        self.assertEqual(len(result), 2)  # unchanged

    def test_unknown_key_skips_filter(self):
        unknown_key = {'key': 'Unknown', 'score': 0.0, 'candidates': []}
        events = [self._make_event('C#4'), self._make_event('F4')]
        result = self.mapper.filter_by_key(events, unknown_key)
        self.assertEqual(len(result), 2)

    def test_empty_events_returns_empty(self):
        result = self.mapper.filter_by_key([], self.c_major_key)
        self.assertEqual(result, [])

    # --- integration with process() ---

    def test_key_filter_runs_inside_process(self):
        """process() should remove out-of-key notes when apply_key_filter=True."""
        # Pitch data for C4 (261.63 Hz) and C#4 (277.18 Hz) — C# is not in C major
        pitch_data = [
            {'time': 0.0,  'frequency': 261.63, 'confidence': 0.9},   # C4
            {'time': 0.05, 'frequency': 261.63, 'confidence': 0.9},
            {'time': 0.1,  'frequency': 261.63, 'confidence': 0.9},
            {'time': 0.2,  'frequency': 329.63, 'confidence': 0.9},   # E4
            {'time': 0.25, 'frequency': 329.63, 'confidence': 0.9},
            {'time': 0.3,  'frequency': 392.00, 'confidence': 0.9},   # G4
            {'time': 0.4,  'frequency': 277.18, 'confidence': 0.9},   # C#4 — out of key
            {'time': 0.45, 'frequency': 277.18, 'confidence': 0.9},
        ]
        mapper = NoteMapper(confidence_threshold=0.0, apply_key_filter=True)
        result = mapper.process(pitch_data)
        notes = [e['note'] for e in result]
        self.assertNotIn('C#4', notes)
        self.assertIn('C4', notes)

    def test_key_filter_off_preserves_all_notes(self):
        """apply_key_filter=False means out-of-key notes pass through untouched."""
        pitch_data = [
            {'time': 0.0,  'frequency': 261.63, 'confidence': 0.9},   # C4
            {'time': 0.1,  'frequency': 277.18, 'confidence': 0.9},   # C#4
        ]
        mapper = NoteMapper(confidence_threshold=0.0, apply_key_filter=False)
        result = mapper.process(pitch_data)
        notes = [e['note'] for e in result]
        self.assertIn('C4', notes)
        self.assertIn('C#4', notes)

    def test_map_notes_convenience_respects_flag(self):
        pitch_data = [
            {'time': 0.0,  'frequency': 261.63, 'confidence': 0.9},
            {'time': 0.05, 'frequency': 261.63, 'confidence': 0.9},
            {'time': 0.1,  'frequency': 329.63, 'confidence': 0.9},
            {'time': 0.15, 'frequency': 329.63, 'confidence': 0.9},
            {'time': 0.2,  'frequency': 392.00, 'confidence': 0.9},
            {'time': 0.3,  'frequency': 277.18, 'confidence': 0.9},   # C#4
        ]
        with_filter    = map_notes(pitch_data, confidence_threshold=0.0, apply_key_filter=True)
        without_filter = map_notes(pitch_data, confidence_threshold=0.0, apply_key_filter=False)
        notes_with    = [e['note'] for e in with_filter]
        notes_without = [e['note'] for e in without_filter]
        self.assertNotIn('C#4', notes_with)
        self.assertIn('C#4', notes_without)


class TestExistingTestsUnaffected(unittest.TestCase):
    """Smoke-check that the new default (apply_key_filter=True) doesn't break
    the original test fixtures, which are built from diatonic notes anyway."""

    def test_original_synthetic_data_still_produces_4_events(self):
        """The original test_note_mapper fixture uses C4/E4/G4 — all diatonic.
        Key filter should not remove any of them."""
        mapper = NoteMapper(confidence_threshold=0.85)  # default — filter ON
        synthetic = [
            {'time': 0.0,  'frequency': 261.63, 'confidence': 0.90},
            {'time': 0.05, 'frequency': 262.0,  'confidence': 0.88},
            {'time': 0.10, 'frequency': 260.5,  'confidence': 0.92},
            {'time': 0.15, 'frequency': 261.0,  'confidence': 0.70},  # below threshold
            {'time': 0.20, 'frequency': 329.63, 'confidence': 0.95},
            {'time': 0.25, 'frequency': 330.0,  'confidence': 0.93},
            {'time': 0.30, 'frequency': 392.0,  'confidence': 0.96},
            {'time': 0.45, 'frequency': 262.0,  'confidence': 0.91},
            {'time': 0.50, 'frequency': 261.5,  'confidence': 0.89},
        ]
        result = mapper.process(synthetic)
        self.assertEqual(len(result), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
