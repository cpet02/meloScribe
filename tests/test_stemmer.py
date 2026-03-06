"""
Unit tests for Phase 1: Stemmer
"""

import unittest
import os
import tempfile
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.stemmer import Stemmer, isolate_vocals


class TestStemmer(unittest.TestCase):
    """Test cases for the Stemmer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent.parent / "samples"
        self.sample_mp3 = self.test_data_dir / "sample.mp3"
        
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp(prefix="test_stemmer_")
        
        # Initialize stemmer
        self.stemmer = Stemmer()
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            self.stemmer.separate_vocals("nonexistent.mp3")
        
        # Test non-MP3 file (create a dummy file)
        dummy_file = Path(self.temp_dir) / "dummy.txt"
        dummy_file.write_text("not an mp3")
        
        with self.assertRaises(ValueError):
            self.stemmer.separate_vocals(str(dummy_file))
    
    @unittest.skipUnless(os.path.exists(str(Path(__file__).parent.parent / "samples" / "sample.mp3")),
                        "sample.mp3 not found in samples directory")
    def test_vocal_isolation(self):
        """Test vocal isolation with actual sample file."""
        # Skip if sample file doesn't exist
        if not self.sample_mp3.exists():
            self.skipTest(f"Sample file not found: {self.sample_mp3}")
        
        # Test isolation
        vocals_path = self.stemmer.separate_vocals(
            str(self.sample_mp3),
            output_dir=self.temp_dir,
            force_reprocess=True
        )
        
        # Verify output
        self.assertIsInstance(vocals_path, str)
        self.assertTrue(os.path.exists(vocals_path))
        
        # Check it's a WAV file
        self.assertTrue(vocals_path.lower().endswith('.wav'))
        
        # Verify file is not empty
        file_size = os.path.getsize(vocals_path)
        self.assertGreater(file_size, 1000)  # At least 1KB
        
        print(f"\nTest successful: Created {vocals_path} ({file_size} bytes)")
    
    @unittest.skipUnless(os.path.exists(str(Path(__file__).parent.parent / "samples" / "sample.mp3")),
                        "sample.mp3 not found in samples directory")
    def test_duration_check(self):
        """Test that output duration is reasonable compared to input."""
        import librosa
        
        if not self.sample_mp3.exists():
            self.skipTest(f"Sample file not found: {self.sample_mp3}")
        
        # Get input duration
        input_duration, _ = librosa.load(str(self.sample_mp3), sr=None, mono=False)
        if len(input_duration.shape) > 1:
            input_duration = input_duration.shape[1] / librosa.get_samplerate(str(self.sample_mp3))
        else:
            input_duration = len(input_duration) / librosa.get_samplerate(str(self.sample_mp3))
        
        # Isolate vocals
        vocals_path = self.stemmer.separate_vocals(
            str(self.sample_mp3),
            output_dir=self.temp_dir,
            force_reprocess=True
        )
        
        # Get output duration
        output_duration, _ = librosa.load(vocals_path, sr=None, mono=False)
        if len(output_duration.shape) > 1:
            output_duration = output_duration.shape[1] / librosa.get_samplerate(vocals_path)
        else:
            output_duration = len(output_duration) / librosa.get_samplerate(vocals_path)
        
        # Check duration similarity (within 2 seconds as specified)
        duration_diff = abs(input_duration - output_duration)
        self.assertLessEqual(duration_diff, 2.0,
                           f"Duration difference too large: {duration_diff:.2f}s "
                           f"(input: {input_duration:.2f}s, output: {output_duration:.2f}s)")
        
        print(f"\nDuration check passed: Input={input_duration:.2f}s, "
              f"Output={output_duration:.2f}s, Diff={duration_diff:.2f}s")
    
    def test_caching_behavior(self):
        """Test that caching works (skip reprocessing if output exists)."""
        # This test requires sample.mp3
        if not self.sample_mp3.exists():
            self.skipTest(f"Sample file not found: {self.sample_mp3}")
        
        # First run - should process
        vocals_path1 = self.stemmer.separate_vocals(
            str(self.sample_mp3),
            output_dir=self.temp_dir,
            force_reprocess=True
        )
        
        # Get modification time
        mtime1 = os.path.getmtime(vocals_path1)
        
        # Second run without force - should use cached
        vocals_path2 = self.stemmer.separate_vocals(
            str(self.sample_mp3),
            output_dir=self.temp_dir,
            force_reprocess=False
        )
        
        # Should be same file
        self.assertEqual(vocals_path1, vocals_path2)
        
        # Modification time should be unchanged
        mtime2 = os.path.getmtime(vocals_path2)
        self.assertEqual(mtime1, mtime2)
        
        # Third run with force - should reprocess
        vocals_path3 = self.stemmer.separate_vocals(
            str(self.sample_mp3),
            output_dir=self.temp_dir,
            force_reprocess=True
        )
        
        # Modification time should be newer
        mtime3 = os.path.getmtime(vocals_path3)
        self.assertGreater(mtime3, mtime1)
        
        print("\nCaching behavior test passed")
    
    def test_convenience_function(self):
        """Test the isolate_vocals convenience function."""
        if not self.sample_mp3.exists():
            self.skipTest(f"Sample file not found: {self.sample_mp3}")
        
        vocals_path = isolate_vocals(
            str(self.sample_mp3),
            output_dir=self.temp_dir,
            force_reprocess=True
        )
        
        self.assertTrue(os.path.exists(vocals_path))
        self.assertTrue(vocals_path.lower().endswith('.wav'))
        
        print(f"\nConvenience function test passed: {vocals_path}")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)