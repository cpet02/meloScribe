"""
Phase 3: Note Mapping and Segmentation
Converts raw pitch detection to note events with filtering and grouping.
"""

import librosa
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class NoteEvent:
    """Represents a single note event with timing and confidence."""
    note: str
    start_time: float
    end_time: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'note': self.note,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'confidence': self.confidence
        }


class NoteMapper:
    """Maps raw pitch detection to note events with filtering and segmentation."""

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, confidence_threshold: float = 0.85, transpose: int = 0):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {confidence_threshold}")
        self.confidence_threshold = confidence_threshold
        self.transpose = transpose

    def hz_to_note_name(self, frequency: float) -> str:
        if frequency <= 0:
            raise ValueError(f"Frequency must be positive, got {frequency}")
        note_number = librosa.hz_to_midi(frequency)
        return librosa.midi_to_note(note_number, unicode=False)

    def _transpose_note(self, note: str, semitones: int) -> str:
        midi = librosa.note_to_midi(note)
        return librosa.midi_to_note(int(midi) + semitones, unicode=False)

    def filter_by_confidence(self, pitch_data: List[Dict[str, float]]) -> List[Dict[str, float]]:
        return [
            frame for frame in pitch_data
            if frame['confidence'] >= self.confidence_threshold
        ]

    def segment_notes(self, pitch_data: List[Dict[str, float]]) -> List[NoteEvent]:
        if not pitch_data:
            return []

        frames_with_notes = []
        for frame in pitch_data:
            try:
                note_name = self.hz_to_note_name(frame['frequency'])
                frames_with_notes.append({
                    'time': frame['time'],
                    'note': note_name,
                    'confidence': frame['confidence']
                })
            except ValueError:
                continue

        if not frames_with_notes:
            return []

        note_events = []
        current_note = frames_with_notes[0]['note']
        current_start = frames_with_notes[0]['time']
        current_confidences = [frames_with_notes[0]['confidence']]

        for i in range(1, len(frames_with_notes)):
            frame = frames_with_notes[i]
            time_gap = frame['time'] - frames_with_notes[i-1]['time']

            if frame['note'] != current_note or time_gap > 0.1:
                avg_confidence = np.mean(current_confidences)
                note_events.append(NoteEvent(
                    note=current_note,
                    start_time=current_start,
                    end_time=frames_with_notes[i-1]['time'],
                    confidence=avg_confidence
                ))
                current_note = frame['note']
                current_start = frame['time']
                current_confidences = [frame['confidence']]
            else:
                current_confidences.append(frame['confidence'])

        avg_confidence = np.mean(current_confidences)
        note_events.append(NoteEvent(
            note=current_note,
            start_time=current_start,
            end_time=frames_with_notes[-1]['time'],
            confidence=avg_confidence
        ))

        return note_events

    def process(self, pitch_data: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        filtered_data = self.filter_by_confidence(pitch_data)
        note_events = self.segment_notes(filtered_data)
        result = [event.to_dict() for event in note_events]

        if self.transpose:
            result = [
                {**event, 'note': self._transpose_note(event['note'], self.transpose)}
                for event in result
            ]

        return result


# All 24 keys and their pitch classes
_SCALES = {
    'C major':  ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    'G major':  ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
    'D major':  ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
    'A major':  ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
    'E major':  ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
    'B major':  ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#'],
    'F# major': ['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'F'],
    'F major':  ['F', 'G', 'A', 'A#', 'C', 'D', 'E'],
    'Bb major': ['A#', 'C', 'D', 'D#', 'F', 'G', 'A'],
    'Eb major': ['D#', 'F', 'G', 'G#', 'A#', 'C', 'D'],
    'Ab major': ['G#', 'A#', 'C', 'C#', 'D#', 'F', 'G'],
    'Db major': ['C#', 'D#', 'F', 'F#', 'G#', 'A#', 'C'],
    'A minor':  ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'E minor':  ['E', 'F#', 'G', 'A', 'B', 'C', 'D'],
    'B minor':  ['B', 'C#', 'D', 'E', 'F#', 'G', 'A'],
    'F# minor': ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E'],
    'C# minor': ['C#', 'D#', 'E', 'F#', 'G#', 'A', 'B'],
    'D minor':  ['D', 'E', 'F', 'G', 'A', 'A#', 'C'],
    'G minor':  ['G', 'A', 'A#', 'C', 'D', 'D#', 'F'],
    'C minor':  ['C', 'D', 'D#', 'F', 'G', 'G#', 'A#'],
    'F minor':  ['F', 'G', 'G#', 'A#', 'C', 'C#', 'D#'],
    'Bb minor': ['A#', 'C', 'C#', 'D#', 'F', 'F#', 'G#'],
    'Eb minor': ['D#', 'F', 'F#', 'G#', 'A#', 'B', 'C#'],
    'Ab minor': ['G#', 'A#', 'B', 'C#', 'D#', 'E', 'F#'],
}


def detect_key(note_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Estimate the key signature from a list of note events.

    Args:
        note_events: List of note event dicts with a 'note' field (e.g. "A#4")

    Returns:
        Dict with:
            'key': best matching key name (e.g. "Bb major")
            'score': fraction of detected notes that fit the key (0-1)
            'candidates': top 3 key matches as list of (key, score) tuples
    """
    if not note_events:
        return {'key': 'Unknown', 'score': 0.0, 'candidates': []}

    def pitch_class(note: str) -> str:
        return ''.join(c for c in note if not c.isdigit()).strip()

    pitch_classes = [pitch_class(e['note']) for e in note_events if e.get('note')]
    total = len(pitch_classes)

    if total == 0:
        return {'key': 'Unknown', 'score': 0.0, 'candidates': []}

    scores = {}
    for key_name, scale in _SCALES.items():
        matches = sum(1 for pc in pitch_classes if pc in scale)
        scores[key_name] = matches / total

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_key, best_score = ranked[0]

    return {
        'key': best_key,
        'score': best_score,
        'candidates': ranked[:3]
    }


def map_notes(pitch_data: List[Dict[str, float]],
              confidence_threshold: float = 0.85,
              transpose: int = 0) -> List[Dict[str, Any]]:
    """
    Convenience function for note mapping.

    Args:
        pitch_data: Raw pitch detection output
        confidence_threshold: Minimum confidence threshold
        transpose: Semitones to transpose (default 0, use 9 for alto sax Eb)

    Returns:
        List of note event dictionaries
    """
    mapper = NoteMapper(confidence_threshold=confidence_threshold, transpose=transpose)
    return mapper.process(pitch_data)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Convert pitch data to note events")
    parser.add_argument("input", help="Input JSON file with pitch data")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")
    parser.add_argument("--threshold", "-t", type=float, default=0.85)
    parser.add_argument("--transpose", type=int, default=0,
                        help="Semitones to transpose (default 0, use 9 for alto sax)")

    args = parser.parse_args()

    try:
        with open(args.input, 'r') as f:
            pitch_data = json.load(f)

        note_events = map_notes(pitch_data, args.threshold, transpose=args.transpose)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(note_events, f, indent=2)
            print(f"Saved {len(note_events)} note events to {args.output}")
        else:
            print(f"\nProcessed {len(pitch_data)} pitch frames")
            print(f"Result: {len(note_events)} note events")

            if note_events:
                print("\nFirst 5 note events:")
                for i, event in enumerate(note_events[:5]):
                    duration = event['end_time'] - event['start_time']
                    print(f"  {i}: {event['note']} ({duration:.3f}s) "
                          f"[{event['start_time']:.3f}-{event['end_time']:.3f}] "
                          f"conf={event['confidence']:.3f}")

    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.input}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)