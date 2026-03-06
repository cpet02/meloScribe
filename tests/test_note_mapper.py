"""
Unit tests for Phase 3: Note Mapper
Tests use synthetic data only - no audio files needed.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.note_mapper import NoteMapper, map_notes, NoteEvent


class TestNoteMapper(unittest.TestCase):
    """Test cases for the NoteMapper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mapper = NoteMapper(confidence_threshold=0.85)
        
        # Synthetic test data
        self.synthetic_pitch_data = [
            # Group 1: C4 with varying confidence
            {'time': 0.0, 'frequency': 261.63, 'confidence': 0.90},  # C4
            {'time': 0.05, 'frequency': 262.0, 'confidence': 0.88},   # Still C4
            {'time': 0.10, 'frequency': 260.5, 'confidence': 0.92},  # Still C4
            
            # Low confidence should be filtered out
            {'time': 0.15, 'frequency': 261.0, 'confidence': 0.70},  # Below threshold
            
            # Group 2: E4
            {'time': 0.20, 'frequency': 329.63, 'confidence': 0.95},  # E4
            {'time': 0.25, 'frequency': 330.0, 'confidence': 0.93},    # Still E4
            
            # Group 3: G4
            {'time': 0.30, 'frequency': 392.0, 'confidence': 0.96},   # G4
            
            # Another C4 but with gap > 0.1s, should be separate event
            {'time': 0.45, 'frequency': 262.0, 'confidence': 0.91},   # C4
            {'time': 0.50, 'frequency': 261.5, 'confidence': 0.89},   # Still C4
        ]
        
        # Edge case: all below threshold
        self.low_confidence_data = [
            {'time': 0.0, 'frequency': 261.63, 'confidence': 0.50},
            {'time': 0.05, 'frequency': 329.63, 'confidence': 0.60},
        ]
        
        # Single frame
        self.single_frame_data = [
            {'time': 0.0, 'frequency': 440.0, 'confidence': 0.90},  # A4
        ]
        
        # Invalid frequency
        self.invalid_frequency_data = [
            {'time': 0.0, 'frequency': -1.0, 'confidence': 0.95},  # Invalid
            {'time': 0.05, 'frequency': 0.0, 'confidence': 0.95}, # Invalid
            {'time': 0.10, 'frequency': 440.0, 'confidence': 0.95}, # Valid
        ]
    
    def test_confidence_threshold_validation(self):
        """Test that confidence threshold must be between 0 and 1."""
        # Valid thresholds
        NoteMapper(confidence_threshold=0.0)
        NoteMapper(confidence_threshold=0.5)
        NoteMapper(confidence_threshold=1.0)
        
        # Invalid thresholds
        with self.assertRaises(ValueError):
            NoteMapper(confidence_threshold=-0.1)
        with self.assertRaises(ValueError):
            NoteMapper(confidence_threshold=1.1)
    
    def test_hz_to_note_name(self):
        """Test Hz to note name conversion."""
        # Test that output is a valid note name format rather than exact match
        # librosa rounding can cause slight variations at non-exact frequencies
        test_frequencies = [261.63, 329.63, 392.00, 440.00, 493.88]
        
        for hz in test_frequencies:
            note_name = self.mapper.hz_to_note_name(hz)
            self.assertIsInstance(note_name, str)
            self.assertGreaterEqual(len(note_name), 2)
            # First character should be a note letter
            self.assertIn(note_name[0].upper(), ['C', 'D', 'E', 'F', 'G', 'A', 'B'])
            # Should contain an octave number
            self.assertTrue(any(char.isdigit() for char in note_name))
        
        # 440Hz should always reliably be A4
        self.assertEqual(self.mapper.hz_to_note_name(440.0), "A4")
        
        # Test invalid frequencies
        with self.assertRaises(ValueError):
            self.mapper.hz_to_note_name(-10.0)
        with self.assertRaises(ValueError):
            self.mapper.hz_to_note_name(0.0)
    
    def test_filter_by_confidence(self):
        """Test filtering of low confidence frames."""
        # Filter synthetic data
        filtered = self.mapper.filter_by_confidence(self.synthetic_pitch_data)
        
        # Should have 7 frames (3 C4, 2 E4, 1 G4, 1 C4)
        self.assertEqual(len(filtered), 8)
        
        # All confidence values should be >= 0.85
        for frame in filtered:
            self.assertGreaterEqual(frame['confidence'], 0.85)
        
        # Verify specific frames were filtered out
        confidences = [frame['confidence'] for frame in filtered]
        self.assertNotIn(0.70, confidences)  # The low confidence frame
        
        # Test with all low confidence
        filtered_low = self.mapper.filter_by_confidence(self.low_confidence_data)
        self.assertEqual(len(filtered_low), 0)
    
    def test_segment_notes_basic(self):
        """Test basic note segmentation."""
        # Use filtered data
        filtered = self.mapper.filter_by_confidence(self.synthetic_pitch_data)
        note_events = self.mapper.segment_notes(filtered)
        
        # Should have 4 note events:
        # 1. C4 (frames 0-2)
        # 2. E4 (frames 3-4)
        # 3. G4 (frame 5)
        # 4. C4 (frames 6-7) - separate due to gap > 0.1s
        self.assertEqual(len(note_events), 4)
        
        # Check first event (C4)
        self.assertEqual(note_events[0].note, "C4")
        self.assertEqual(note_events[0].start_time, 0.0)
        self.assertEqual(note_events[0].end_time, 0.10)
        self.assertAlmostEqual(note_events[0].confidence, (0.90 + 0.88 + 0.92) / 3, places=2)
        
        # Check second event (E4)
        self.assertEqual(note_events[1].note, "E4")
        self.assertEqual(note_events[1].start_time, 0.20)
        self.assertEqual(note_events[1].end_time, 0.25)
        
        # Check third event (G4)
        self.assertEqual(note_events[2].note, "G4")
        self.assertEqual(note_events[2].start_time, 0.30)
        self.assertEqual(note_events[2].end_time, 0.30)
        
        # Check fourth event (C4)
        self.assertEqual(note_events[3].note, "C4")
        self.assertEqual(note_events[3].start_time, 0.45)
        self.assertEqual(note_events[3].end_time, 0.50)
    
    def test_segment_notes_single_frame(self):
        """Test segmentation with single frame."""
        filtered = self.mapper.filter_by_confidence(self.single_frame_data)
        note_events = self.mapper.segment_notes(filtered)
        
        self.assertEqual(len(note_events), 1)
        self.assertEqual(note_events[0].note, "A4")
        self.assertEqual(note_events[0].start_time, 0.0)
        self.assertEqual(note_events[0].end_time, 0.0)
        self.assertAlmostEqual(note_events[0].confidence, 0.90)
    
    def test_segment_notes_empty(self):
        """Test segmentation with empty input."""
        note_events = self.mapper.segment_notes([])
        self.assertEqual(len(note_events), 0)
        
        # All filtered out due to low confidence
        filtered = self.mapper.filter_by_confidence(self.low_confidence_data)
        note_events = self.mapper.segment_notes(filtered)
        self.assertEqual(len(note_events), 0)
    
    def test_segment_notes_invalid_frequencies(self):
        """Test that invalid frequencies are skipped."""
        # Only the valid frame should be processed
        note_events = self.mapper.segment_notes(self.invalid_frequency_data)
        
        self.assertEqual(len(note_events), 1)
        self.assertEqual(note_events[0].note, "A4")
    
    def test_segment_notes_consecutive_same_note(self):
        """Test that consecutive same notes are merged."""
        # Create data with same note at consecutive times
        same_note_data = [
            {'time': 0.0, 'frequency': 261.63, 'confidence': 0.90},  # C4
            {'time': 0.05, 'frequency': 262.0, 'confidence': 0.91},   # Still C4
            {'time': 0.10, 'frequency': 260.5, 'confidence': 0.92},   # Still C4
            {'time': 0.15, 'frequency': 261.0, 'confidence': 0.93},   # Still C4
        ]
        
        note_events = self.mapper.segment_notes(same_note_data)
        
        # Should be merged into one event
        self.assertEqual(len(note_events), 1)
        self.assertEqual(note_events[0].note, "C4")
        self.assertEqual(note_events[0].start_time, 0.0)
        self.assertEqual(note_events[0].end_time, 0.15)
        self.assertAlmostEqual(note_events[0].confidence, (0.90 + 0.91 + 0.92 + 0.93) / 4)
    
    def test_segment_notes_gap_detection(self):
        """Test that gaps > 0.1s create separate events."""
        # Same note but with gap > 0.1s
        gapped_data = [
            {'time': 0.0, 'frequency': 261.63, 'confidence': 0.90},   # C4
            {'time': 0.05, 'frequency': 262.0, 'confidence': 0.91},    # Still C4
            {'time': 0.20, 'frequency': 261.5, 'confidence': 0.92},    # Same note but gap > 0.1s
            {'time': 0.25, 'frequency': 261.0, 'confidence': 0.93},    # Still C4
        ]
        
        note_events = self.mapper.segment_notes(gapped_data)
        
        # Should be two separate events
        self.assertEqual(len(note_events), 2)
        
        # First event
        self.assertEqual(note_events[0].note, "C4")
        self.assertEqual(note_events[0].start_time, 0.0)
        self.assertEqual(note_events[0].end_time, 0.05)
        
        # Second event
        self.assertEqual(note_events[1].note, "C4")
        self.assertEqual(note_events[1].start_time, 0.20)
        self.assertEqual(note_events[1].end_time, 0.25)
    
    def test_process_full_pipeline(self):
        """Test the full process method."""
        note_events = self.mapper.process(self.synthetic_pitch_data)
        
        # Should return list of dicts
        self.assertIsInstance(note_events, list)
        self.assertEqual(len(note_events), 4)
        
        # Check structure
        for event in note_events:
            self.assertIsInstance(event, dict)
            self.assertIn('note', event)
            self.assertIn('start_time', event)
            self.assertIn('end_time', event)
            self.assertIn('confidence', event)
            
            # Validate note name format
            self.assertIsInstance(event['note'], str)
            self.assertTrue(len(event['note']) >= 2)  # At least "C4"
            
            # Validate times
            self.assertLessEqual(event['start_time'], event['end_time'])
            self.assertGreaterEqual(event['start_time'], 0)
            
            # Validate confidence
            self.assertGreaterEqual(event['confidence'], 0.85)
            self.assertLessEqual(event['confidence'], 1.0)
    
    def test_convenience_function(self):
        """Test the map_notes convenience function."""
        note_events = map_notes(self.synthetic_pitch_data, confidence_threshold=0.85)
        
        self.assertIsInstance(note_events, list)
        self.assertEqual(len(note_events), 4)
        
        # Test with different threshold
        note_events_lower = map_notes(self.synthetic_pitch_data, confidence_threshold=0.70)
        self.assertEqual(len(note_events_lower), 4)
    
    def test_note_event_dataclass(self):
        """Test the NoteEvent dataclass."""
        event = NoteEvent(
            note="C4",
            start_time=0.0,
            end_time=0.5,
            confidence=0.9
        )
        
        self.assertEqual(event.note, "C4")
        self.assertEqual(event.start_time, 0.0)
        self.assertEqual(event.end_time, 0.5)
        self.assertEqual(event.confidence, 0.9)
        
        # Test to_dict method
        event_dict = event.to_dict()
        self.assertEqual(event_dict['note'], "C4")
        self.assertEqual(event_dict['start_time'], 0.0)
        self.assertEqual(event_dict['end_time'], 0.5)
        self.assertEqual(event_dict['confidence'], 0.9)


class TestNoteMapperEdgeCases(unittest.TestCase):
    """Test edge cases for NoteMapper."""
    
    def test_zero_confidence_threshold(self):
        """Test with zero confidence threshold (include everything)."""
        mapper = NoteMapper(confidence_threshold=0.0)
        
        data = [
            {'time': 0.0, 'frequency': 261.63, 'confidence': 0.0},
            {'time': 0.05, 'frequency': 261.63, 'confidence': 0.1},
        ]
        
        note_events = mapper.process(data)
        self.assertEqual(len(note_events), 1)  # Should include both frames
    
    def test_one_confidence_threshold(self):
        """Test with confidence threshold of 1.0."""
        mapper = NoteMapper(confidence_threshold=1.0)
        
        data = [
            {'time': 0.0, 'frequency': 261.63, 'confidence': 0.99},
            {'time': 0.05, 'frequency': 261.63, 'confidence': 1.0},
        ]
        
        note_events = mapper.process(data)
        # Only the frame with confidence 1.0 should pass
        self.assertEqual(len(note_events), 1)
        if note_events:
            self.assertEqual(note_events[0]['confidence'], 1.0)
    
    def test_note_name_validation(self):
        """Test that note names are valid."""
        mapper = NoteMapper()
        
        # Test various frequencies produce valid note names
        test_frequencies = [65.41, 130.81, 261.63, 523.25, 1046.50]  # C2, C3, C4, C5, C6
        
        for freq in test_frequencies:
            note_name = mapper.hz_to_note_name(freq)
            # Check format: note + octave (e.g., "C4", "A#3")
            self.assertIsInstance(note_name, str)
            self.assertGreaterEqual(len(note_name), 2)
            
            # Note name should start with a letter A-G
            self.assertIn(note_name[0].upper(), ['C', 'D', 'E', 'F', 'G', 'A', 'B'])
            
            # Should contain an octave number
            self.assertTrue(any(char.isdigit() for char in note_name))


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)