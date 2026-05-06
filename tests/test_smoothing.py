"""
Tests for Step 2: Note smoothing — octave outlier removal and flicker merging.
All tests use synthetic data — no audio files required.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.note_mapper import NoteMapper, map_notes


def make_event(note, start, end, confidence=0.9):
    return {'note': note, 'start_time': start, 'end_time': end, 'confidence': confidence}


class TestRemoveOctaveOutliers(unittest.TestCase):

    def setUp(self):
        self.mapper = NoteMapper(confidence_threshold=0.0, smooth=False)

    # --- core removal ---

    def test_short_outlier_between_same_note_removed(self):
        events = [
            make_event('F#3', 0.0, 0.3),
            make_event('F#7', 0.3, 0.4),   # short, >1 octave from both neighbours
            make_event('F#3', 0.4, 0.7),
        ]
        result = self.mapper.remove_octave_outliers(events)
        self.assertEqual(len(result), 2)
        self.assertFalse(any(e['note'] == 'F#7' for e in result))

    def test_short_outlier_between_different_notes_removed(self):
        events = [
            make_event('A3', 0.0, 0.3),
            make_event('C8', 0.3, 0.38),   # very short, absurdly high
            make_event('B3', 0.38, 0.6),
        ]
        result = self.mapper.remove_octave_outliers(events)
        self.assertEqual(len(result), 2)

    # --- preservation ---

    def test_long_outlier_kept(self):
        """A note >0.15s that looks like an outlier is kept — could be a real leap."""
        events = [
            make_event('F#3', 0.0, 0.3),
            make_event('F#7', 0.3, 0.5),   # long — 0.2s, kept regardless
            make_event('F#3', 0.5, 0.8),
        ]
        result = self.mapper.remove_octave_outliers(events)
        self.assertEqual(len(result), 3)

    def test_outlier_only_one_octave_away_kept(self):
        """Exactly one octave is borderline and should be kept (> 12, not >= 12)."""
        events = [
            make_event('A3', 0.0, 0.3),
            make_event('A4', 0.3, 0.38),   # exactly 12 semitones — not an outlier
            make_event('A3', 0.38, 0.6),
        ]
        result = self.mapper.remove_octave_outliers(events)
        self.assertEqual(len(result), 3)

    def test_first_and_last_events_never_removed(self):
        """Edge notes have only one neighbour — never removed."""
        events = [
            make_event('C8', 0.0, 0.05),   # suspiciously high but it's first
            make_event('A3', 0.05, 0.3),
            make_event('C8', 0.3, 0.35),   # suspiciously high but it's last
        ]
        result = self.mapper.remove_octave_outliers(events)
        self.assertEqual(len(result), 3)

    def test_fewer_than_3_events_unchanged(self):
        events = [make_event('A3', 0.0, 0.3), make_event('B3', 0.3, 0.6)]
        result = self.mapper.remove_octave_outliers(events)
        self.assertEqual(len(result), 2)

    def test_empty_input(self):
        self.assertEqual(self.mapper.remove_octave_outliers([]), [])


class TestMergeFlickers(unittest.TestCase):

    def setUp(self):
        self.mapper = NoteMapper(confidence_threshold=0.0, smooth=False)

    # --- core merging ---

    def test_same_note_tiny_gap_merged(self):
        events = [
            make_event('A4', 0.0, 0.2, confidence=0.9),
            make_event('A4', 0.24, 0.5, confidence=0.88),  # gap = 0.04s
        ]
        result = self.mapper.merge_flickers(events, max_gap=0.08)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['start_time'], 0.0)
        self.assertAlmostEqual(result[0]['end_time'], 0.5)
        self.assertAlmostEqual(result[0]['confidence'], (0.9 + 0.88) / 2, places=5)

    def test_three_flickers_all_merged(self):
        events = [
            make_event('G4', 0.0,  0.1),
            make_event('G4', 0.14, 0.25),  # gap 0.04s
            make_event('G4', 0.29, 0.4),   # gap 0.04s
        ]
        result = self.mapper.merge_flickers(events, max_gap=0.08)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['start_time'], 0.0)
        self.assertAlmostEqual(result[0]['end_time'], 0.4)

    # --- preservation ---

    def test_real_rest_not_merged(self):
        events = [
            make_event('A4', 0.0, 0.2),
            make_event('A4', 0.5, 0.8),   # gap = 0.3s — intentional rest
        ]
        result = self.mapper.merge_flickers(events, max_gap=0.08)
        self.assertEqual(len(result), 2)

    def test_different_notes_not_merged(self):
        events = [
            make_event('A4', 0.0,  0.2),
            make_event('B4', 0.22, 0.4),   # tiny gap but different note
        ]
        result = self.mapper.merge_flickers(events, max_gap=0.08)
        self.assertEqual(len(result), 2)

    def test_different_octaves_not_merged(self):
        """A4 and A3 are different notes — not merged even with a tiny gap."""
        events = [
            make_event('A4', 0.0,  0.2),
            make_event('A3', 0.22, 0.4),
        ]
        result = self.mapper.merge_flickers(events, max_gap=0.08)
        self.assertEqual(len(result), 2)

    def test_empty_input(self):
        self.assertEqual(self.mapper.merge_flickers([]), [])

    def test_single_event_unchanged(self):
        events = [make_event('C4', 0.0, 0.3)]
        result = self.mapper.merge_flickers(events)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['note'], 'C4')


class TestSmoothingInProcess(unittest.TestCase):
    """Integration — verify smoothing runs (or doesn't) inside process()."""

    def _make_pitch(self, hz, t, conf=0.9):
        return {'time': t, 'frequency': hz, 'confidence': conf}

    def test_smooth_on_by_default(self):
        mapper = NoteMapper(confidence_threshold=0.0)
        self.assertTrue(mapper.smooth)

    def test_process_removes_outlier_with_smooth_on(self):
        # C4=261.63, C7=2093.0 (outlier), E4=329.63
        pitch_data = [
            self._make_pitch(261.63, 0.0),
            self._make_pitch(261.63, 0.05),
            self._make_pitch(261.63, 0.1),
            self._make_pitch(2093.0, 0.2),   # C7 — outlier, short
            self._make_pitch(329.63, 0.3),
            self._make_pitch(329.63, 0.35),
        ]
        mapper = NoteMapper(confidence_threshold=0.0, apply_key_filter=False, smooth=True)
        result = mapper.process(pitch_data)
        notes = [e['note'] for e in result]
        self.assertNotIn('C7', notes)
        self.assertIn('C4', notes)
        self.assertIn('E4', notes)

    def test_process_merges_flicker_with_smooth_on(self):
        # Two identical C4 runs with a tiny gap between them
        pitch_data = [
            self._make_pitch(261.63, 0.0),
            self._make_pitch(261.63, 0.05),
            # gap of ~0.06s
            self._make_pitch(261.63, 0.11),
            self._make_pitch(261.63, 0.16),
        ]
        mapper = NoteMapper(confidence_threshold=0.0, apply_key_filter=False, smooth=True)
        result = mapper.process(pitch_data)
        c4_events = [e for e in result if e['note'] == 'C4']
        self.assertEqual(len(c4_events), 1)

    def test_smooth_off_preserves_outlier(self):
        pitch_data = [
            self._make_pitch(261.63, 0.0),
            self._make_pitch(261.63, 0.05),
            self._make_pitch(2093.0, 0.2),   # C7
            self._make_pitch(329.63, 0.3),
        ]
        mapper = NoteMapper(confidence_threshold=0.0, apply_key_filter=False, smooth=False)
        result = mapper.process(pitch_data)
        notes = [e['note'] for e in result]
        self.assertIn('C7', notes)

    def test_map_notes_smooth_flag_respected(self):
        pitch_data = [
            self._make_pitch(261.63, 0.0),
            self._make_pitch(261.63, 0.05),
            self._make_pitch(2093.0, 0.2),
            self._make_pitch(329.63, 0.3),
        ]
        with_smooth    = map_notes(pitch_data, confidence_threshold=0.0,
                                   apply_key_filter=False, smooth=True)
        without_smooth = map_notes(pitch_data, confidence_threshold=0.0,
                                   apply_key_filter=False, smooth=False)
        notes_with    = [e['note'] for e in with_smooth]
        notes_without = [e['note'] for e in without_smooth]
        self.assertNotIn('C7', notes_with)
        self.assertIn('C7', notes_without)


if __name__ == '__main__':
    unittest.main(verbosity=2)
