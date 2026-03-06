"""
Phase 4: Lyric Alignment
Aligns lyrics from LRC files with note events based on timing.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class LyricAligner:
    """Aligns lyrics with note events based on LRC file timestamps."""
    
    # Regex pattern for LRC timestamp lines: [mm:ss.xx]text
    LRC_PATTERN = re.compile(r'\[(\d{2}):(\d{2})(?:\.(\d{2}))?\](.*)')
    
    def parse_lrc_file(self, lrc_path: str) -> List[Tuple[float, str]]:
        """
        Parse an LRC file into timestamped lyrics.
        
        Args:
            lrc_path: Path to LRC file
            
        Returns:
            List of (timestamp, lyric) pairs
            
        Raises:
            FileNotFoundError: If LRC file doesn't exist
            ValueError: If file format is invalid
        """
        lrc_path = Path(lrc_path).resolve()
        if not lrc_path.exists():
            raise FileNotFoundError(f"LRC file not found: {lrc_path}")
        
        if lrc_path.suffix.lower() != '.lrc':
            raise ValueError(f"File must be LRC format. Got: {lrc_path.suffix}")
        
        timestamped_lyrics = []
        
        with open(lrc_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and metadata lines (e.g., [ti:Title])
                if not line or line.startswith('[ti:') or line.startswith('[ar:') \
                   or line.startswith('[al:') or line.startswith('[by:'):
                    continue
                
                # Parse timestamp line
                match = self.LRC_PATTERN.match(line)
                if match:
                    minutes = int(match.group(1))
                    seconds = int(match.group(2))
                    milliseconds = match.group(3)
                    
                    # Convert to seconds
                    timestamp = minutes * 60 + seconds
                    if milliseconds:
                        timestamp += int(milliseconds) / 100.0  # .xx means hundredths
                    
                    lyric = match.group(4).strip()
                    
                    # Skip empty lyrics
                    if lyric:
                        timestamped_lyrics.append((timestamp, lyric))
        
        if not timestamped_lyrics:
            raise ValueError(f"No valid lyrics found in {lrc_path}")
        
        # Sort by timestamp
        timestamped_lyrics.sort(key=lambda x: x[0])
        
        return timestamped_lyrics
    
    def parse_lrc_content(self, lrc_content: str) -> List[Tuple[float, str]]:
        """
        Parse LRC content from string instead of file.
        
        Args:
            lrc_content: String containing LRC formatted lyrics
            
        Returns:
            List of (timestamp, lyric) pairs
        """
        timestamped_lyrics = []
        
        for line in lrc_content.strip().split('\n'):
            line = line.strip()
            
            # Skip empty lines and metadata
            if not line or line.startswith('[ti:') or line.startswith('[ar:') \
               or line.startswith('[al:') or line.startswith('[by:'):
                continue
            
            # Parse timestamp line
            match = self.LRC_PATTERN.match(line)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                milliseconds = match.group(3)
                
                # Convert to seconds
                timestamp = minutes * 60 + seconds
                if milliseconds:
                    timestamp += int(milliseconds) / 100.0
                
                lyric = match.group(4).strip()
                
                if lyric:
                    timestamped_lyrics.append((timestamp, lyric))
        
        # Sort by timestamp
        timestamped_lyrics.sort(key=lambda x: x[0])
        
        return timestamped_lyrics
    
    def align_lyrics(self, note_events: List[Dict[str, Any]], 
                     lrc_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Align lyrics with note events.
        
        Args:
            note_events: List of note event dicts
            lrc_path: Optional path to LRC file
            
        Returns:
            Same note events with added 'lyric' field (null if no match or no LRC)
        """
        # If no LRC file, set all lyrics to null
        if lrc_path is None:
            for event in note_events:
                event['lyric'] = None
            return note_events
        
        # Parse LRC file
        try:
            timestamped_lyrics = self.parse_lrc_file(lrc_path)
        except Exception as e:
            # If LRC parsing fails, fall back to null lyrics
            print(f"Warning: Failed to parse LRC file: {e}")
            for event in note_events:
                event['lyric'] = None
            return note_events
        
        # Align each note event with lyrics
        for event in note_events:
            event['lyric'] = None
            
            # Find lyric whose timestamp window contains the note's start time
            note_time = event['start_time']
            
            # Look for matching lyric
            for i, (lyric_time, lyric_text) in enumerate(timestamped_lyrics):
                # Check if note falls in this lyric's window
                # Window is from this lyric's timestamp to next lyric's timestamp
                next_time = timestamped_lyrics[i + 1][0] if i + 1 < len(timestamped_lyrics) else float('inf')
                
                if lyric_time <= note_time < next_time:
                    event['lyric'] = lyric_text
                    break
        
        return note_events
    
    def align_lyrics_from_content(self, note_events: List[Dict[str, Any]], 
                                  lrc_content: str) -> List[Dict[str, Any]]:
        """
        Align lyrics from LRC content string.
        
        Args:
            note_events: List of note event dicts
            lrc_content: LRC formatted string
            
        Returns:
            Same note events with added 'lyric' field
        """
        timestamped_lyrics = self.parse_lrc_content(lrc_content)
        
        for event in note_events:
            event['lyric'] = None
            
            note_time = event['start_time']
            
            for i, (lyric_time, lyric_text) in enumerate(timestamped_lyrics):
                next_time = timestamped_lyrics[i + 1][0] if i + 1 < len(timestamped_lyrics) else float('inf')
                
                if lyric_time <= note_time < next_time:
                    event['lyric'] = lyric_text
                    break
        
        return note_events


def align_lyrics_to_notes(note_events: List[Dict[str, Any]], 
                          lrc_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function for lyric alignment.
    
    Args:
        note_events: List of note event dicts
        lrc_path: Optional path to LRC file
        
    Returns:
        Same note events with added 'lyric' field
    """
    aligner = LyricAligner()
    return aligner.align_lyrics(note_events, lrc_path)


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(description="Align lyrics from LRC file with note events")
    parser.add_argument("notes", help="Input JSON file with note events")
    parser.add_argument("--lrc", "-l", help="Path to LRC file (optional)")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")
    
    args = parser.parse_args()
    
    try:
        # Load note events
        with open(args.notes, 'r') as f:
            note_events = json.load(f)
        
        # Align lyrics
        aligned_events = align_lyrics_to_notes(note_events, args.lrc)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(aligned_events, f, indent=2)
            print(f"Saved {len(aligned_events)} aligned events to {args.output}")
        else:
            # Print summary
            print(f"\nAligned {len(aligned_events)} note events")
            
            # Count matched lyrics
            matched = sum(1 for event in aligned_events if event['lyric'] is not None)
            print(f"Lyrics matched: {matched}")
            
            if aligned_events:
                print("\nFirst 5 aligned events:")
                for i, event in enumerate(aligned_events[:5]):
                    lyric = event['lyric'] or "null"
                    print(f"  {i}: {event['note']} [{event['start_time']:.3f}s] -> '{lyric}'")
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.notes}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
