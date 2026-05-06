"""
Tests for Step 3: Beat grid detection and beat alignment annotation.
All tests use mocks — no audio files required.
"""

import sys
import unittest
import unittest.mock as mock
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.beat_tracker import BeatTracker, get_beat_grid


class TestBeatTrackerOutput(unittest.TestCase):

    def _make_tracker(self):
        return BeatTracker(subdivisions=4)

    def _mock_librosa(self, bpm=120.0, beat_frames=None):
        """Return context managers patching librosa.load and librosa.beat.beat_track."""
        if beat_frames is None:
            beat_frames = np.array([10, 20, 30, 40, 50])
        beat_times = beat_frames * (512 / 22050)  # default hop/sr

        load_patch = mock.patch('librosa.load',
                                return_value=(np.zeros(44100), 22050))
        beat_patch = mock.patch('librosa.beat.beat_track',
                                return_value=(np.array([bpm]), beat_frames))
        frames_patch = mock.patch('librosa.frames_to_time',
                                  return_value=beat_times)
        return load_patch, beat_patch, frames_patch

    # --- output structure ---

    def test_output_has_required_keys(self):
        tracker = self._make_tracker()
        load_p, beat_p, frames_p = self._mock_librosa()
        with mock.patch('pathlib.Path.exists', return_value=True), \
             load_p, beat_p, frames_p:
            result = tracker.detect_beats('fake.wav')
        self.assertIn('bpm', result)
        self.assertIn('beat_times', result)
        self.assertIn('subdivisions', result)

    def test_bpm_is_float(self):
        tracker = self._make_tracker()
        load_p, beat_p, frames_p = self._mock_librosa(bpm=98.4)
        with mock.patch('pathlib.Path.exists', return_value=True), \
             load_p, beat_p, frames_p:
            result = tracker.detect_beats('fake.wav')
        self.assertIsInstance(result['bpm'], float)
        self.assertAlmostEqual(result['bpm'], 98.4, places=1)

    def test_beat_times_is_list(self):
        tracker = self._make_tracker()
        load_p, beat_p, frames_p = self._mock_librosa()
        with mock.patch('pathlib.Path.exists', return_value=True), \
             load_p, beat_p, frames_p:
            result = tracker.detect_beats('fake.wav')
        self.assertIsInstance(result['beat_times'], list)
        self.assertGreater(len(result['beat_times']), 0)

    def test_subdivisions_is_list_and_longer_than_beats(self):
        tracker = self._make_tracker()
        load_p, beat_p, frames_p = self._mock_librosa()
        with mock.patch('pathlib.Path.exists', return_value=True), \
             load_p, beat_p, frames_p:
            result = tracker.detect_beats('fake.wav')
        self.assertIsInstance(result['subdivisions'], list)
        # With 4 subdivisions per beat interval, subs should be >= beat count
        self.assertGreaterEqual(len(result['subdivisions']),
                                len(result['beat_times']))

    def test_subdivisions_are_sorted(self):
        tracker = self._make_tracker()
        load_p, beat_p, frames_p = self._mock_librosa()
        with mock.patch('pathlib.Path.exists', return_value=True), \
             load_p, beat_p, frames_p:
            result = tracker.detect_beats('fake.wav')
        subs = result['subdivisions']
        self.assertEqual(subs, sorted(subs))

    def test_file_not_found_raises(self):
        tracker = self._make_tracker()
        with self.assertRaises(FileNotFoundError):
            tracker.detect_beats('definitely_does_not_exist.wav')

    # --- subdivision building ---

    def test_subdivision_count(self):
        """4 subdivisions per interval across N-1 intervals = 4*(N-1) + 1 points."""
        tracker = BeatTracker(subdivisions=4)
        beat_times = np.array([0.0, 0.5, 1.0, 1.5])  # 3 intervals
        subs = tracker._build_subdivisions(beat_times, 120.0)
        # 4 * 3 intervals + 1 final beat = 13
        self.assertEqual(len(subs), 13)

    def test_subdivisions_bracket_beats(self):
        tracker = BeatTracker(subdivisions=2)
        beat_times = np.array([0.0, 1.0])
        subs = tracker._build_subdivisions(beat_times, 60.0)
        self.assertIn(0.0, subs)
        self.assertIn(0.5, subs)  # midpoint
        self.assertIn(1.0, subs)

    def test_single_beat_returns_safely(self):
        tracker = self._make_tracker()
        beat_times = np.array([0.5])
        subs = tracker._build_subdivisions(beat_times, 120.0)
        self.assertEqual(subs, [0.5])


class TestBeatAlignmentAnnotation(unittest.TestCase):

    def setUp(self):
        self.tracker = BeatTracker()
        self.beat_grid = {
            'bpm': 120.0,
            'beat_times': [0.0, 0.5, 1.0, 1.5, 2.0],
            'subdivisions': [0.0, 0.125, 0.25, 0.375,
                             0.5, 0.625, 0.75, 0.875,
                             1.0, 1.125, 1.25, 1.375,
                             1.5, 1.625, 1.75, 1.875, 2.0],
        }

    def _event(self, note, t):
        return {'note': note, 'start_time': t, 'end_time': t + 0.1, 'confidence': 0.9}

    def test_on_beat_note_aligned(self):
        events = [self._event('C4', 0.0)]
        result = self.tracker.annotate_beat_alignment(events, self.beat_grid, tolerance=0.08)
        self.assertTrue(result[0]['beat_aligned'])

    def test_near_beat_within_tolerance_aligned(self):
        events = [self._event('C4', 0.06)]  # 0.06s from beat at 0.0 — within 0.08
        result = self.tracker.annotate_beat_alignment(events, self.beat_grid, tolerance=0.08)
        self.assertTrue(result[0]['beat_aligned'])

    def test_between_subdivisions_not_aligned(self):
        events = [self._event('C4', 0.3)]   # 0.3s — between subs at 0.25 and 0.375
        result = self.tracker.annotate_beat_alignment(events, self.beat_grid, tolerance=0.04)
        self.assertFalse(result[0]['beat_aligned'])

    def test_beat_aligned_field_added_to_all_events(self):
        events = [self._event('C4', t) for t in [0.0, 0.3, 0.5, 0.9]]
        result = self.tracker.annotate_beat_alignment(events, self.beat_grid)
        for e in result:
            self.assertIn('beat_aligned', e)
            self.assertIsInstance(e['beat_aligned'], bool)

    def test_empty_subdivisions_all_false(self):
        events = [self._event('C4', 0.0)]
        result = self.tracker.annotate_beat_alignment(events, {'subdivisions': []})
        self.assertFalse(result[0]['beat_aligned'])

    def test_empty_events_returns_empty(self):
        result = self.tracker.annotate_beat_alignment([], self.beat_grid)
        self.assertEqual(result, [])

    def test_annotation_does_not_alter_other_fields(self):
        events = [self._event('D4', 0.5)]
        result = self.tracker.annotate_beat_alignment(events, self.beat_grid)
        self.assertEqual(result[0]['note'], 'D4')
        self.assertAlmostEqual(result[0]['start_time'], 0.5)
        self.assertAlmostEqual(result[0]['confidence'], 0.9)


class TestGetBeatGridConvenience(unittest.TestCase):

    def test_convenience_function_returns_dict(self):
        beat_times = np.array([0.0, 0.5, 1.0])
        with mock.patch('pathlib.Path.exists', return_value=True), \
             mock.patch('librosa.load', return_value=(np.zeros(22050), 22050)), \
             mock.patch('librosa.beat.beat_track',
                        return_value=(np.array([120.0]), np.array([0, 10, 20]))), \
             mock.patch('librosa.frames_to_time', return_value=beat_times):
            result = get_beat_grid('fake.wav', subdivisions=4)
        self.assertIn('bpm', result)
        self.assertIn('beat_times', result)
        self.assertIn('subdivisions', result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
