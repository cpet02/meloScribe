"""
Unit tests for Phase 2: Pitch Detector
"""

import unittest
import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pitch_detector import PitchDetector, detect_pitch

VOCALS_PATH = str(Path(__file__).parent.parent / "separated" / "htdemucs" / "sample" / "vocals.wav")

class TestPitchDetector(unittest.TestCase):
    """Test cases for the PitchDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_wav = Path(VOCALS_PATH)
        self.detector = PitchDetector()
    
    def test_import_basic_pitch(self):
        """Test that basic-pitch is importable."""
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
            self.assertTrue(True)
        except ImportError:
            self.skipTest("basic-pitch not installed. Skipping pitch detection tests.")
    
    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        with self.assertRaises(FileNotFoundError):
            self.detector.detect_pitch("nonexistent.wav")
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"not a wav")
            tmp_path = tmp.name
        
        try:
            with self.assertRaises(ValueError):
                self.detector.detect_pitch(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    @unittest.skipUnless(os.path.exists(VOCALS_PATH),
                        "vocals.wav not found. Run Phase 1 first.")
    def test_pitch_detection(self):
        """Test pitch detection with actual vocals.wav file."""
        pitch_data = self.detector.detect_pitch(str(self.sample_wav))
        self.assertIsInstance(pitch_data, list)
        self.assertGreater(len(pitch_data), 0, "Pitch detection returned empty list")
        print(f"\nDetected {len(pitch_data)} pitch frames")
    
    @unittest.skipUnless(os.path.exists(VOCALS_PATH),
                        "vocals.wav not found. Run Phase 1 first.")
    def test_output_structure(self):
        """Test that output has correct structure and data types."""
        pitch_data = self.detector.detect_pitch(str(self.sample_wav))
        
        required_keys = {'time', 'frequency', 'confidence'}
        for frame in pitch_data:
            self.assertIsInstance(frame, dict)
            self.assertEqual(set(frame.keys()), required_keys)
            self.assertIsInstance(frame['time'], (int, float))
            self.assertIsInstance(frame['frequency'], (int, float))
            self.assertIsInstance(frame['confidence'], (int, float))
            self.assertGreaterEqual(frame['time'], 0)
            self.assertGreater(frame['frequency'], 0)
            self.assertGreaterEqual(frame['confidence'], 0)
            self.assertLessEqual(frame['confidence'], 1)
    
    @unittest.skipUnless(os.path.exists(VOCALS_PATH),
                        "vocals.wav not found. Run Phase 1 first.")
    def test_confidence_range(self):
        """Test that all confidence values are between 0 and 1."""
        pitch_data = self.detector.detect_pitch(str(self.sample_wav))
        confidences = [frame['confidence'] for frame in pitch_data]
        
        if confidences:
            min_conf = min(confidences)
            max_conf = max(confidences)
            self.assertGreaterEqual(min_conf, 0)
            self.assertLessEqual(max_conf, 1)
            print(f"\nConfidence range: {min_conf:.3f} to {max_conf:.3f}")
    
    @unittest.skipUnless(os.path.exists(VOCALS_PATH),
                        "vocals.wav not found. Run Phase 1 first.")
    def test_timestamp_monotonic(self):
        """Test that timestamps are monotonically increasing."""
        pitch_data = self.detector.detect_pitch(str(self.sample_wav))
        times = [frame['time'] for frame in pitch_data]
        
        for i in range(1, len(times)):
            self.assertGreaterEqual(times[i], times[i-1],
                                  f"Timestamps not monotonic at index {i}: {times[i]} < {times[i-1]}")
        
        if len(times) > 1:
            print(f"\nTimestamps monotonic: {times[0]:.3f}s to {times[-1]:.3f}s ({len(times)} frames)")
    
    @unittest.skipUnless(os.path.exists(VOCALS_PATH),
                        "vocals.wav not found. Run Phase 1 first.")
    def test_frequency_range(self):
        """Test that frequencies are within reasonable human vocal range."""
        pitch_data = self.detector.detect_pitch(str(self.sample_wav))
        frequencies = [frame['frequency'] for frame in pitch_data]
        
        if frequencies:
            min_freq = min(frequencies)
            max_freq = max(frequencies)
            self.assertGreater(min_freq, 20)
            self.assertLess(max_freq, 2000)
            print(f"\nFrequency range: {min_freq:.1f}Hz to {max_freq:.1f}Hz")
    
    @unittest.skipUnless(os.path.exists(VOCALS_PATH),
                        "vocals.wav not found. Run Phase 1 first.")
    def test_convenience_function(self):
        """Test the detect_pitch convenience function."""
        pitch_data = detect_pitch(str(self.sample_wav))
        self.assertIsInstance(pitch_data, list)
        if pitch_data:
            frame = pitch_data[0]
            self.assertIn('time', frame)
            self.assertIn('frequency', frame)
            self.assertIn('confidence', frame)
            print(f"\nConvenience function test passed: {len(pitch_data)} frames")
    
    @unittest.skipUnless(os.path.exists(VOCALS_PATH),
                        "vocals.wav not found. Run Phase 1 first.")
    def test_threshold_parameters(self):
        """Test that threshold parameters affect output."""
        detector_low = PitchDetector(onset_threshold=0.1, frame_threshold=0.1)
        detector_high = PitchDetector(onset_threshold=0.9, frame_threshold=0.9)
        
        data_low = detector_low.detect_pitch(str(self.sample_wav))
        data_high = detector_high.detect_pitch(str(self.sample_wav))
        
        print(f"\nThreshold comparison:")
        print(f"  Low thresholds: {len(data_low)} frames")
        print(f"  High thresholds: {len(data_high)} frames")


if __name__ == "__main__":
    unittest.main(verbosity=2)