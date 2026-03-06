"""
Unit tests for Phase 4: Lyric Aligner
Tests use synthetic data only - no files needed.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.lyric_aligner import LyricAligner, align_lyrics_to_notes


class TestLyricAligner(unittest.TestCase):
    """Test cases for the LyricAligner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aligner = LyricAligner()
        
        # Fake note events for testing
        self.fake_note_events = [
            {'note': 'C4', 'start_time': 0.0, 'end_time': 0.5, 'confidence': 0.9},
            {'note': 'E4', 'start_time': 1.0, 'end_time': 1.5, 'confidence': 0.95},
            {'note': 'G4', 'start_time': 2.5, 'end_time': 3.0, 'confidence': 0.92},
            {'note': 'C4', 'start_time': 3.5, 'end_time': 4.0, 'confidence': 0.88},
            {'note': 'A4', 'start_time': 5.0, 'end_time': 5.5, 'confidence': 0.96},
        ]
        
        # Fake LRC content for testing
        self.fake_lrc_content = """
[ti:Test Song]
[ar:Test Artist]
[al:Test Album]
[by:Test Writer]
[00:00.00]Hello
[00:01.00]world
[00:02.50]this
[00:03.50]is
[00:05.00]a test
"""
        
        # Fake LRC with gaps
        self.fake_lrc_with_gaps = """
[00:00.00]Start
[00:02.00]Middle
[00:04.00]End
"""
        
        # Fake LRC with milliseconds
        self.fake_lrc_with_ms = """
[00:00.50]Half second
[00:01.25]One and quarter
[00:02.75]Two and three quarters
"""
        
        # Fake LRC with empty lyrics
        self.fake_lrc_empty = """
[00:00.00]
[00:01.00]Real lyric
[00:02.00]
"""
    
    def test_parse_lrc_content(self):
        """Test parsing LRC content from string."""
        parsed = self.aligner.parse_lrc_content(self.fake_lrc_content)
        
        # Should have 5 lyrics (skipping metadata lines)
        self.assertEqual(len(parsed), 5)
        
        # Check parsed values
        expected = [
            (0.0, "Hello"),
            (1.0, "world"),
            (2.5, "this"),
            (3.5, "is"),
            (5.0, "a test"),
        ]
        
        for i, (timestamp, lyric) in enumerate(parsed):
            self.assertEqual(timestamp, expected[i][0])
            self.assertEqual(lyric, expected[i][1])
        
        # Should be sorted by timestamp
        timestamps = [t for t, _ in parsed]
        self.assertEqual(timestamps, sorted(timestamps))
    
    def test_parse_lrc_content_with_gaps(self):
        """Test parsing LRC with gaps."""
        parsed = self.aligner.parse_lrc_content(self.fake_lrc_with_gaps)
        
        self.assertEqual(len(parsed), 3)
        
        expected = [
            (0.0, "Start"),
            (2.0, "Middle"),
            (4.0, "End"),
        ]
        
        for i, (timestamp, lyric) in enumerate(parsed):
            self.assertEqual(timestamp, expected[i][0])
            self.assertEqual(lyric, expected[i][1])
    
    def test_parse_lrc_content_with_milliseconds(self):
        """Test parsing LRC with milliseconds."""
        parsed = self.aligner.parse_lrc_content(self.fake_lrc_with_ms)
        
        self.assertEqual(len(parsed), 3)
        
        expected = [
            (0.5, "Half second"),
            (1.25, "One and quarter"),
            (2.75, "Two and three quarters"),
        ]
        
        for i, (timestamp, lyric) in enumerate(parsed):
            self.assertEqual(timestamp, expected[i][0])
            self.assertEqual(lyric, expected[i][1])
    
    def test_parse_lrc_content_empty_lyrics(self):
        """Test parsing LRC with empty lyrics."""
        parsed = self.aligner.parse_lrc_content(self.fake_lrc_empty)
        
        # Should only have 1 lyric (the non-empty one)
        self.assertEqual(len(parsed), 1)
        
        self.assertEqual(parsed[0][0], 1.0)
        self.assertEqual(parsed[0][1], "Real lyric")
    
    def test_align_lyrics_from_content(self):
        """Test aligning lyrics from content string."""
        aligned = self.aligner.align_lyrics_from_content(
            self.fake_note_events, 
            self.fake_lrc_content
        )
        
        # Should have same number of events
        self.assertEqual(len(aligned), len(self.fake_note_events))
        
        # Check each event has lyric field
        for event in aligned:
            self.assertIn('lyric', event)
            
            # Original fields should be preserved
            self.assertIn('note', event)
            self.assertIn('start_time', event)
            self.assertIn('end_time', event)
            self.assertIn('confidence', event)
        
        # Check specific alignments
        # Note at 0.0s -> "Hello" (timestamp 0.0)
        self.assertEqual(aligned[0]['lyric'], "Hello")
        
        # Note at 1.0s -> "world" (timestamp 1.0)
        self.assertEqual(aligned[1]['lyric'], "world")
        
        # Note at 2.5s -> "this" (timestamp 2.5)
        self.assertEqual(aligned[2]['lyric'], "this")
        
        # Note at 3.5s -> "is" (timestamp 3.5)
        self.assertEqual(aligned[3]['lyric'], "is")
        
        # Note at 5.0s -> "a test" (timestamp 5.0)
        self.assertEqual(aligned[4]['lyric'], "a test")
    
    def test_align_lyrics_from_content_with_gaps(self):
        """Test aligning lyrics with gaps between timestamps."""
        # Create note events that fall in gaps
        gap_note_events = [
            {'note': 'C4', 'start_time': 0.5, 'end_time': 1.0, 'confidence': 0.9},  # Between 0.0 and 2.0
            {'note': 'E4', 'start_time': 3.0, 'end_time': 3.5, 'confidence': 0.95}, # Between 2.0 and 4.0
        ]
        
        aligned = self.aligner.align_lyrics_from_content(
            gap_note_events,
            self.fake_lrc_with_gaps
        )
        
        # Note at 0.5s -> "Start" (window: 0.0-2.0)
        self.assertEqual(aligned[0]['lyric'], "Start")
        
        # Note at 3.0s -> "Middle" (window: 2.0-4.0)
        self.assertEqual(aligned[1]['lyric'], "Middle")
    
    def test_align_lyrics_from_content_no_match(self):
        """Test notes that don't match any lyric window."""
        # Note before first lyric
        early_note = [
            {'note': 'C4', 'start_time': -0.5, 'end_time': 0.0, 'confidence': 0.9},
        ]
        
        aligned = self.aligner.align_lyrics_from_content(
            early_note,
            self.fake_lrc_content
        )
        
        self.assertEqual(aligned[0]['lyric'], None)
        
        # Note before first lyric (second check)
        late_note = [
            {'note': 'C4', 'start_time': -1.0, 'end_time': -0.5, 'confidence': 0.9},
        ]
        
        aligned = self.aligner.align_lyrics_from_content(
            late_note,
            self.fake_lrc_content
        )
        
        self.assertEqual(aligned[0]['lyric'], None)
    
    def test_align_lyrics_from_content_edge_cases(self):
        """Test edge cases in lyric alignment."""
        # Note exactly at lyric timestamp
        exact_note = [
            {'note': 'C4', 'start_time': 1.0, 'end_time': 1.5, 'confidence': 0.9},
        ]
        
        aligned = self.aligner.align_lyrics_from_content(
            exact_note,
            self.fake_lrc_content
        )
        
        self.assertEqual(aligned[0]['lyric'], "world")
        
        # Note between two lyrics
        between_note = [
            {'note': 'C4', 'start_time': 1.5, 'end_time': 2.0, 'confidence': 0.9},
        ]
        
        aligned = self.aligner.align_lyrics_from_content(
            between_note,
            self.fake_lrc_content
        )
        
        # Should match "world" (window: 1.0-2.5)
        self.assertEqual(aligned[0]['lyric'], "world")
    
    def test_align_lyrics_no_lrc(self):
        """Test alignment when no LRC is provided."""
        aligned = self.aligner.align_lyrics(self.fake_note_events, None)
        
        self.assertEqual(len(aligned), len(self.fake_note_events))
        
        # All lyrics should be null
        for event in aligned:
            self.assertEqual(event['lyric'], None)
    
    def test_align_lyrics_empty_content(self):
        """Test alignment with empty LRC content."""
        aligned = self.aligner.align_lyrics_from_content(
            self.fake_note_events,
            ""
        )
        
        # All lyrics should be null
        for event in aligned:
            self.assertEqual(event['lyric'], None)
    
    def test_convenience_function(self):
        """Test the align_lyrics_to_notes convenience function."""
        # Test with content via align_lyrics_from_content
        aligned = self.aligner.align_lyrics_from_content(
            self.fake_note_events,
            self.fake_lrc_content
        )
        
        # Test convenience function with no LRC
        aligned_none = align_lyrics_to_notes(self.fake_note_events, None)
        
        self.assertEqual(len(aligned_none), len(self.fake_note_events))
        for event in aligned_none:
            self.assertEqual(event['lyric'], None)
    
    def test_lrc_pattern_matching(self):
        """Test LRC pattern regex matches correctly."""
        test_lines = [
            "[00:00.00]Hello",          # Standard
            "[01:30.50]World",          # With milliseconds
            "[02:45]No milliseconds",   # Without milliseconds
            "[ti:Title]",               # Metadata (should not match)
            "[ar:Artist]",              # Metadata
            "[00:00.00]",               # Empty lyric
            "[99:99.99]Edge case",      # Edge case (invalid time but pattern matches)
        ]
        
        for line in test_lines:
            match = self.aligner.LRC_PATTERN.match(line)
            
            if line.startswith('[ti:') or line.startswith('[ar:'):
                self.assertIsNone(match, f"Metadata line should not match: {line}")
            elif line == "[00:00.00]":
                # Empty lyric line should match but will be filtered out later
                self.assertIsNotNone(match, f"Empty lyric line should match pattern: {line}")
                if match:
                    self.assertEqual(match.group(4), "")  # Empty lyric
            else:
                self.assertIsNotNone(match, f"Valid LRC line should match: {line}")
    
    def test_timestamp_conversion(self):
        """Test timestamp conversion from LRC format."""
        # Test various timestamp formats
        test_cases = [
            ("00:00", 0.0),      # Minutes:Seconds
            ("01:30", 90.0),     # 1 minute 30 seconds
            ("00:00.50", 0.5),   # With milliseconds (.50 = 0.5 seconds)
            ("01:30.25", 90.25), # 1:30.25 = 90.25 seconds
            ("02:45.75", 165.75), # 2:45.75 = 165.75 seconds
        ]
        
        for time_str, expected_seconds in test_cases:
            # Create a mock LRC line
            line = f"[{time_str}]Test lyric"
            match = self.aligner.LRC_PATTERN.match(line)
            
            self.assertIsNotNone(match)
            
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            milliseconds = match.group(3)
            
            timestamp = minutes * 60 + seconds
            if milliseconds:
                timestamp += int(milliseconds) / 100.0
            
            self.assertEqual(timestamp, expected_seconds)


class TestLyricAlignerEdgeCases(unittest.TestCase):
    """Test edge cases for LyricAligner."""
    
    def setUp(self):
        self.aligner = LyricAligner()
    
    def test_single_lyric(self):
        """Test alignment with only one lyric."""
        note_events = [
            {'note': 'C4', 'start_time': 0.0, 'end_time': 0.5, 'confidence': 0.9},
            {'note': 'E4', 'start_time': 1.0, 'end_time': 1.5, 'confidence': 0.95},
            {'note': 'G4', 'start_time': 2.0, 'end_time': 2.5, 'confidence': 0.92},
        ]
        
        lrc_content = "[00:00.00]Only lyric"
        
        aligned = self.aligner.align_lyrics_from_content(note_events, lrc_content)
        
        # All notes should match the single lyric
        for event in aligned:
            self.assertEqual(event['lyric'], "Only lyric")
    
    def test_notes_at_same_time(self):
        """Test multiple notes at the same start time."""
        note_events = [
            {'note': 'C4', 'start_time': 1.0, 'end_time': 1.5, 'confidence': 0.9},
            {'note': 'E4', 'start_time': 1.0, 'end_time': 1.5, 'confidence': 0.95},  # Same time
            {'note': 'G4', 'start_time': 2.0, 'end_time': 2.5, 'confidence': 0.92},
        ]
        
        lrc_content = """
[00:01.00]First lyric
[00:02.00]Second lyric
"""
        
        aligned = self.aligner.align_lyrics_from_content(note_events, lrc_content)
        
        # Both notes at 1.0s should match "First lyric"
        self.assertEqual(aligned[0]['lyric'], "First lyric")
        self.assertEqual(aligned[1]['lyric'], "First lyric")
        self.assertEqual(aligned[2]['lyric'], "Second lyric")
    
    def test_lyric_window_boundaries(self):
        """Test notes at exact window boundaries."""
        note_events = [
            {'note': 'C4', 'start_time': 1.0, 'end_time': 1.5, 'confidence': 0.9},  # At lyric time
            {'note': 'E4', 'start_time': 2.0, 'end_time': 2.5, 'confidence': 0.95}, # At next lyric time
        ]
        
        lrc_content = """
[00:01.00]Lyric 1
[00:02.00]Lyric 2
"""
        
        aligned = self.aligner.align_lyrics_from_content(note_events, lrc_content)
        
        # Note at 1.0s -> "Lyric 1" (window: 1.0-2.0)
        self.assertEqual(aligned[0]['lyric'], "Lyric 1")
        
        # Note at 2.0s -> "Lyric 2" (window: 2.0-inf)
        self.assertEqual(aligned[1]['lyric'], "Lyric 2")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)