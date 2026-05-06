"""
Phase 3 (supplement): Beat Grid Detection
Detects tempo and beat positions from an audio file using librosa.
Annotates note events with beat alignment for downstream use.
"""

import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


class BeatTracker:
    """Detects beat grid from an audio file and annotates note events."""

    def __init__(self, subdivisions: int = 4):
        """
        Args:
            subdivisions: How many subdivisions per beat to generate.
                          4 = sixteenth notes at any tempo (default).
                          2 = eighth notes only.
        """
        self.subdivisions = subdivisions

    def detect_beats(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect tempo and beat positions from an audio file.

        Args:
            audio_path: Path to audio file (WAV recommended)

        Returns:
            Dict with:
                'bpm':          Estimated tempo in beats per minute (float)
                'beat_times':   Array of beat onset times in seconds
                'subdivisions': Array of subdivision times (beat_times split
                                into self.subdivisions equal parts per beat)

        Raises:
            FileNotFoundError: If audio file does not exist
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"  Detecting beat grid from: {audio_path.name}")

        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Scalar tempo from newer librosa versions may be a 0-d array
        bpm = float(np.atleast_1d(tempo)[0])

        subdivisions = self._build_subdivisions(beat_times, bpm)

        print(f"  Beat grid: {bpm:.1f} BPM, {len(beat_times)} beats detected")

        return {
            'bpm': bpm,
            'beat_times': beat_times.tolist(),
            'subdivisions': subdivisions,
        }

    def _build_subdivisions(self, beat_times: np.ndarray, bpm: float) -> List[float]:
        """
        Interpolate subdivision times between detected beats.

        Args:
            beat_times: Array of beat onset times in seconds
            bpm:        Estimated tempo

        Returns:
            Sorted list of subdivision times covering the full song
        """
        if len(beat_times) < 2:
            return beat_times.tolist()

        subs = []
        for i in range(len(beat_times) - 1):
            start = beat_times[i]
            end   = beat_times[i + 1]
            for s in range(self.subdivisions):
                subs.append(start + (end - start) * s / self.subdivisions)

        # Add the final beat itself
        subs.append(float(beat_times[-1]))
        return sorted(subs)

    def annotate_beat_alignment(self, note_events: List[Dict[str, Any]],
                                beat_grid: Dict[str, Any],
                                tolerance: float = 0.08) -> List[Dict[str, Any]]:
        """
        Add a 'beat_aligned' field to each note event.

        A note is beat-aligned if its start time falls within `tolerance` seconds
        of any subdivision in the beat grid. This is informational only — it does
        not remove or modify any notes.

        Args:
            note_events: List of note event dicts
            beat_grid:   Output of detect_beats()
            tolerance:   Max distance in seconds from a subdivision to count as
                         aligned (default 0.08s — roughly a 16th note at 120 BPM)

        Returns:
            Same list with 'beat_aligned' bool added to each event
        """
        subdivisions = beat_grid.get('subdivisions', [])

        if not subdivisions:
            for event in note_events:
                event['beat_aligned'] = False
            return note_events

        sub_array = np.array(subdivisions)

        for event in note_events:
            t = event['start_time']
            nearest = float(np.min(np.abs(sub_array - t)))
            event['beat_aligned'] = nearest <= tolerance

        aligned = sum(1 for e in note_events if e.get('beat_aligned'))
        total   = len(note_events)
        if total:
            print(f"  Beat alignment: {aligned}/{total} notes on-beat "
                  f"({aligned / total:.0%})")

        return note_events


def get_beat_grid(audio_path: str, subdivisions: int = 4) -> Dict[str, Any]:
    """
    Convenience function — detect beat grid from an audio file.

    Args:
        audio_path:   Path to audio file
        subdivisions: Subdivisions per beat (default 4 = sixteenth notes)

    Returns:
        Dict with 'bpm', 'beat_times', 'subdivisions'
    """
    return BeatTracker(subdivisions=subdivisions).detect_beats(audio_path)


if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description='Detect beat grid from audio file')
    parser.add_argument('input', help='Path to audio file')
    parser.add_argument('--subdivisions', type=int, default=4)
    parser.add_argument('--output', '-o', help='Save beat grid to JSON file')
    args = parser.parse_args()

    try:
        grid = get_beat_grid(args.input, subdivisions=args.subdivisions)
        print(f"\nBPM: {grid['bpm']:.1f}")
        print(f"Beats detected: {len(grid['beat_times'])}")
        print(f"Subdivisions:   {len(grid['subdivisions'])}")
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(grid, f, indent=2)
            print(f"Saved to {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
