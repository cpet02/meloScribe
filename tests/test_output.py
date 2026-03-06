"""
Unit tests for Phase 5: Output Formatter
Tests use synthetic data only - no files needed.
"""

import unittest
import json
import csv
import sys
from io import StringIO
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.output import OutputFormatter, format_output


class TestOutputFormatter(unittest.TestCase):
    """Test cases for the OutputFormatter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Fake note events for testing
        self.fake_note_events = [
            {
                'note': 'C4',
                'start_time': 0.0,
                'end_time': 0.5,
                'confidence': 0.9,
                'lyric': 'Hello'
            },
            {
                'note': 'E4',
                'start_time': 1.0,
                'end_time': 1.5,
                'confidence': 0.95,
                'lyric': 'world'
            },
            {
                'note': 'G4',
                'start_time': 2.5,
                'end_time': 3.0,
                'confidence': 0.92,
                'lyric': None
            },
        ]
        
        # Note events with missing optional field
        self.fake_note_events_minimal = [
            {
                'note': 'A4',
                'start_time': 0.0,
                'end_time': 0.5,
                'confidence': 0.88
                # No lyric field
            }
        ]
        
        # Empty note events
        self.empty_note_events = []
        
        # Initialize formatters
        self.formatter = OutputFormatter(self.fake_note_events)
        self.formatter_minimal = OutputFormatter(self.fake_note_events_minimal)
        self.formatter_empty = OutputFormatter(self.empty_note_events)
    
    def test_format_csv(self):
        """Test CSV formatting."""
        csv_output = self.formatter.format_csv()
        
        # Should not be empty
        self.assertIsInstance(csv_output, str)
        self.assertGreater(len(csv_output), 0)
        
        # Parse CSV to verify structure
        csv_reader = csv.DictReader(StringIO(csv_output))
        
        # Check headers
        expected_headers = ['note', 'start_time', 'end_time', 'confidence', 'lyric']
        self.assertEqual(csv_reader.fieldnames, expected_headers)
        
        # Check rows
        rows = list(csv_reader)
        self.assertEqual(len(rows), len(self.fake_note_events))
        
        # Check first row values
        self.assertEqual(rows[0]['note'], 'C4')
        self.assertEqual(rows[0]['start_time'], '0.0')
        self.assertEqual(rows[0]['end_time'], '0.5')
        self.assertEqual(rows[0]['confidence'], '0.9')
        self.assertEqual(rows[0]['lyric'], 'Hello')
        
        # Check third row has empty string for None lyric
        self.assertIn(rows[2]['lyric'], ['', 'None'])
    
    def test_format_csv_minimal(self):
        """Test CSV formatting with minimal fields."""
        csv_output = self.formatter_minimal.format_csv()
        
        csv_reader = csv.DictReader(StringIO(csv_output))
        rows = list(csv_reader)
        
        # Should have correct headers (no lyric)
        expected_headers = ['note', 'start_time', 'end_time', 'confidence']
        self.assertEqual(csv_reader.fieldnames, expected_headers)
        
        # Should have one row
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['note'], 'A4')
    
    def test_format_csv_empty(self):
        """Test CSV formatting with empty events."""
        csv_output = self.formatter_empty.format_csv()
        
        # Should return empty string
        self.assertEqual(csv_output, "")
    
    def test_format_json(self):
        """Test JSON formatting."""
        json_output = self.formatter.format_json()
        
        # Should be valid JSON
        self.assertIsInstance(json_output, str)
        
        # Parse and verify
        parsed = json.loads(json_output)
        
        # Should match original data structure
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), len(self.fake_note_events))
        
        # Check first event
        self.assertEqual(parsed[0]['note'], 'C4')
        self.assertEqual(parsed[0]['start_time'], 0.0)
        self.assertEqual(parsed[0]['end_time'], 0.5)
        self.assertEqual(parsed[0]['confidence'], 0.9)
        self.assertEqual(parsed[0]['lyric'], 'Hello')
        
        # Check None lyric is preserved as null
        self.assertIsNone(parsed[2]['lyric'])
    
    def test_format_json_compact(self):
        """Test compact JSON formatting."""
        json_output = self.formatter.format_json(indent=None)
        
        # Should be valid JSON
        parsed = json.loads(json_output)
        self.assertEqual(len(parsed), len(self.fake_note_events))
        
        # Should be compact (no newlines)
        self.assertNotIn('\n', json_output)
    
    def test_format_json_minimal(self):
        """Test JSON formatting with minimal fields."""
        json_output = self.formatter_minimal.format_json()
        parsed = json.loads(json_output)
        
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]['note'], 'A4')
        self.assertNotIn('lyric', parsed[0])  # Missing field should not be in JSON
    
    def test_format_json_empty(self):
        """Test JSON formatting with empty events."""
        json_output = self.formatter_empty.format_json()
        parsed = json.loads(json_output)
        
        self.assertEqual(parsed, [])
    
    def test_format_table(self):
        """Test table formatting."""
        # Mock tabulate import for testing
        try:
            from tabulate import tabulate
            TABULATE_AVAILABLE = True
        except ImportError:
            TABULATE_AVAILABLE = False
        
        table_output = self.formatter.format_table()
        
        self.assertIsInstance(table_output, str)
        
        if TABULATE_AVAILABLE:
            # Should contain expected content
            self.assertIn('C4', table_output)
            self.assertIn('E4', table_output)
            self.assertIn('G4', table_output)
            self.assertIn('Hello', table_output)
            self.assertIn('world', table_output)
            
            # Should contain table headers
            self.assertIn('Note', table_output)
            self.assertIn('Start (s)', table_output)
            self.assertIn('End (s)', table_output)
            self.assertIn('Confidence', table_output)
            self.assertIn('Lyric', table_output)
        else:
            # Should return error message
            self.assertIn('tabulate module not installed', table_output)
    
    def test_format_table_minimal(self):
        """Test table formatting with minimal fields."""
        table_output = self.formatter_minimal.format_table()
        
        self.assertIsInstance(table_output, str)
        
        try:
            from tabulate import tabulate
            # Should contain the note
            self.assertIn('A4', table_output)
        except ImportError:
            pass  # Skip content check if tabulate not available
    
    def test_format_table_empty(self):
        """Test table formatting with empty events."""
        table_output = self.formatter_empty.format_table()
        
        self.assertIsInstance(table_output, str)
        self.assertIn('No note events', table_output)
    
    def test_output_method_table(self):
        """Test output method with table format."""
        result = self.formatter.output("table")
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_output_method_csv(self):
        """Test output method with CSV format."""
        result = self.formatter.output("csv")
        
        self.assertIsInstance(result, str)
        
        # Should be valid CSV
        csv_reader = csv.DictReader(StringIO(result))
        rows = list(csv_reader)
        self.assertEqual(len(rows), len(self.fake_note_events))
    
    def test_output_method_json(self):
        """Test output method with JSON format."""
        result = self.formatter.output("json")
        
        self.assertIsInstance(result, str)
        
        # Should be valid JSON
        parsed = json.loads(result)
        self.assertEqual(len(parsed), len(self.fake_note_events))
    
    def test_output_method_invalid_format(self):
        """Test output method with invalid format."""
        with self.assertRaises(ValueError):
            self.formatter.output("invalid_format")
    
    def test_output_method_case_insensitive(self):
        """Test output method is case insensitive."""
        # Should handle uppercase
        result1 = self.formatter.output("JSON")
        result2 = self.formatter.output("json")
        
        # Both should be valid JSON
        parsed1 = json.loads(result1)
        parsed2 = json.loads(result2)
        self.assertEqual(parsed1, parsed2)
    
    def test_convenience_function(self):
        """Test the format_output convenience function."""
        # Test table format
        result = format_output(self.fake_note_events, "table")
        self.assertIsInstance(result, str)
        
        # Test CSV format
        result = format_output(self.fake_note_events, "csv")
        self.assertIsInstance(result, str)
        
        # Test JSON format
        result = format_output(self.fake_note_events, "json")
        self.assertIsInstance(result, str)
        
        # Test with invalid format
        with self.assertRaises(ValueError):
            format_output(self.fake_note_events, "invalid")
    
    def test_print_methods(self):
        """Test the print methods."""
        # These methods should not raise exceptions
        try:
            self.formatter.print_table()
            self.formatter.print_csv()
            self.formatter.print_json()
        except Exception as e:
            self.fail(f"Print method raised exception: {e}")
    
    def test_csv_field_order(self):
        """Test that CSV fields are in consistent order."""
        # Create events with fields in different order
        unordered_events = [
            {
                'confidence': 0.9,
                'start_time': 0.0,
                'lyric': 'Hello',
                'end_time': 0.5,
                'note': 'C4'
            }
        ]
        
        formatter = OutputFormatter(unordered_events)
        csv_output = formatter.format_csv()
        
        # Parse CSV
        csv_reader = csv.DictReader(StringIO(csv_output))
        
        # Fields should be in preferred order
        expected_order = ['note', 'start_time', 'end_time', 'confidence', 'lyric']
        self.assertEqual(csv_reader.fieldnames, expected_order)
    
    def test_csv_extra_fields(self):
        """Test CSV with extra fields beyond standard ones."""
        events_with_extra = [
            {
                'note': 'C4',
                'start_time': 0.0,
                'end_time': 0.5,
                'confidence': 0.9,
                'lyric': 'Hello',
                'extra_field': 'extra_value'
            }
        ]
        
        formatter = OutputFormatter(events_with_extra)
        csv_output = formatter.format_csv()
        
        csv_reader = csv.DictReader(StringIO(csv_output))
        
        # Should include all fields, with standard ones first
        expected_fields = ['note', 'start_time', 'end_time', 'confidence', 'lyric', 'extra_field']
        self.assertEqual(csv_reader.fieldnames, expected_fields)
        
        rows = list(csv_reader)
        self.assertEqual(rows[0]['extra_field'], 'extra_value')


class TestOutputFormatterEdgeCases(unittest.TestCase):
    """Test edge cases for OutputFormatter."""
    
    def test_none_values_in_csv(self):
        """Test CSV formatting with None values."""
        events_with_nones = [
            {
                'note': None,
                'start_time': 0.0,
                'end_time': None,
                'confidence': 0.9,
                'lyric': None
            }
        ]
        
        formatter = OutputFormatter(events_with_nones)
        csv_output = formatter.format_csv()
        
        csv_reader = csv.DictReader(StringIO(csv_output))
        rows = list(csv_reader)
        
        # None values should be empty strings in CSV
        self.assertEqual(rows[0]['note'], '')
        self.assertEqual(rows[0]['end_time'], '')
        self.assertEqual(rows[0]['lyric'], '')
    
    def test_special_characters(self):
        """Test formatting with special characters."""
        events_special = [
            {
                'note': 'C#4',
                'start_time': 0.0,
                'end_time': 0.5,
                'confidence': 0.9,
                'lyric': 'Hello, "world" & <test>'
            }
        ]
        
        formatter = OutputFormatter(events_special)
        
        # Test CSV with special characters
        csv_output = formatter.format_csv()
        self.assertIn('C#4', csv_output)
        csv_reader = csv.DictReader(StringIO(csv_output))
        rows = list(csv_reader)
        self.assertEqual(rows[0]['lyric'], 'Hello, "world" & <test>')
        
        # Test JSON with special characters
        json_output = formatter.format_json()
        self.assertIn('C#4', json_output)
        parsed = json.loads(json_output)
        self.assertEqual(parsed[0]['lyric'], 'Hello, "world" & <test>')
        
        # Test table with special characters
        table_output = formatter.format_table()
        try:
            from tabulate import tabulate
            self.assertIn('C#4', table_output)
        except ImportError:
            pass
    
    def test_unicode_characters(self):
        """Test formatting with Unicode characters."""
        events_unicode = [
            {
                'note': 'A4',
                'start_time': 0.0,
                'end_time': 0.5,
                'confidence': 0.9,
                'lyric': 'Hello 世界 🌍'
            }
        ]
        
        formatter = OutputFormatter(events_unicode)
        
        # JSON should handle Unicode
        json_output = formatter.format_json()
        self.assertIn('Hello 世界 🌍', json_output)
        
        # Should not raise UnicodeEncodeError
        try:
            csv_output = formatter.format_csv()
            table_output = formatter.format_table()
        except UnicodeEncodeError as e:
            self.fail(f"Unicode handling failed: {e}")
    
    def test_very_long_values(self):
        """Test formatting with very long values."""
        long_lyric = 'A' * 1000  # Very long lyric
        
        events_long = [
            {
                'note': 'C4',
                'start_time': 0.0,
                'end_time': 0.5,
                'confidence': 0.9,
                'lyric': long_lyric
            }
        ]
        
        formatter = OutputFormatter(events_long)
        
        # Should not raise errors
        try:
            csv_output = formatter.format_csv()
            json_output = formatter.format_json()
            table_output = formatter.format_table()
            
            # Should contain at least part of the long value
            self.assertIn('C4', csv_output)
            self.assertIn('C4', json_output)
            
        except Exception as e:
            self.fail(f"Failed with long values: {e}")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)