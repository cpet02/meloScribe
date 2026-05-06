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

    def __init__(self, confidence_threshold: float = 0.85, transpose: int = 0,
                 apply_key_filter: bool = True, smooth: bool = True):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {confidence_threshold}")
        self.confidence_threshold = confidence_threshold
        self.transpose = transpose
        self.apply_key_filter = apply_key_filter
        self.smooth = smooth

    def hz_to_note_name(self, frequency: float) -> str:
        if frequency <= 0:
            raise ValueError(f"Frequency must be positive, got {frequency}")
        note_number = librosa.hz_to_midi(frequency)
        return librosa.midi_to_note(note_number, unicode=False)

    def _transpose_note(self, note: str, semitones: int) -> str:
        midi = librosa.note_to_midi(note)
        return librosa.midi_to_note(int(midi) + semitones, unicode=False)

    def filter_by_key(self, note_events: List[Dict[str, Any]],
                      key_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Remove notes whose pitch class falls outside the detected key's scale.

        Only filters when key detection is confident (score >= 0.6). Below that
        threshold the key estimate is too uncertain to use as a hard filter, so
        the full note list is returned unchanged.

        Args:
            note_events: List of note event dicts (must have 'note' field)
            key_result:  Output of detect_key() — needs 'key' and 'score' fields

        Returns:
            Filtered list of note events
        """
        key_name = key_result.get('key', 'Unknown')
        score = key_result.get('score', 0.0)

        # Don't filter if key is unknown or detection confidence is too low
        if key_name == 'Unknown' or score < 0.6 or key_name not in _SCALES:
            return note_events

        scale = set(_SCALES[key_name])

        def pitch_class(note: str) -> str:
            return ''.join(c for c in note if not c.isdigit()).strip()

        kept, removed = [], 0
        for event in note_events:
            pc = pitch_class(event.get('note', ''))
            if pc in scale:
                kept.append(event)
            else:
                removed += 1

        if removed:
            print(f"  Key filter ({key_name}): removed {removed} out-of-key note(s)")

        return kept

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

    def remove_octave_outliers(self, note_events: List[Dict[str, Any]],
                               max_duration: float = 0.15) -> List[Dict[str, Any]]:
        """
        Remove short notes that are more than one octave away from both neighbours.

        These are almost always spurious overtone hits — a detector briefly locks onto
        a harmonic rather than the fundamental. Only removes the note when it is short
        (below max_duration) AND an outlier relative to both the note before and after
        it. Isolated notes at the start/end of the sequence are never removed.

        Args:
            note_events:   List of note event dicts (must have 'note', 'start_time',
                           'end_time' fields)
            max_duration:  Maximum duration in seconds for a note to be considered a
                           candidate for removal (default 0.15s)

        Returns:
            Filtered list with outlier notes removed
        """
        if len(note_events) < 3:
            return note_events

        def midi(note: str) -> int:
            try:
                return librosa.note_to_midi(note)
            except Exception:
                return None

        kept = []
        for i, event in enumerate(note_events):
            if i == 0 or i == len(note_events) - 1:
                kept.append(event)
                continue

            duration = event['end_time'] - event['start_time']
            if duration >= max_duration:
                kept.append(event)
                continue

            mid  = midi(event['note'])
            prev = midi(note_events[i - 1]['note'])
            nxt  = midi(note_events[i + 1]['note'])

            if mid is None or prev is None or nxt is None:
                kept.append(event)
                continue

            # Outlier = more than 12 semitones (one octave) from BOTH neighbours
            if abs(mid - prev) > 12 and abs(mid - nxt) > 12:
                continue  # drop it

            kept.append(event)

        removed = len(note_events) - len(kept)
        if removed:
            print(f"  Smoothing: removed {removed} octave outlier(s)")

        return kept

    def merge_flickers(self, note_events: List[Dict[str, Any]],
                       max_gap: float = 0.08) -> List[Dict[str, Any]]:
        """
        Merge consecutive same-pitch notes separated by a very short gap.

        A gap under max_gap between two identical notes is almost certainly a
        detection dropout rather than an intentional rest. Merges them into one
        event spanning both, averaging confidence.

        Args:
            note_events: List of note event dicts
            max_gap:     Maximum gap in seconds to bridge (default 0.08s)

        Returns:
            List with flicker pairs merged
        """
        if not note_events:
            return []

        def pitch_class_and_octave(note: str) -> str:
            return note  # full note name comparison (e.g. "A4" == "A4")

        merged = [dict(note_events[0])]  # copy first event

        for event in note_events[1:]:
            prev = merged[-1]
            gap = event['start_time'] - prev['end_time']

            if (pitch_class_and_octave(event['note']) == pitch_class_and_octave(prev['note'])
                    and 0 <= gap <= max_gap):
                # Extend previous event to cover this one, average confidence
                prev['end_time'] = event['end_time']
                prev['confidence'] = float(np.mean([prev['confidence'], event['confidence']]))
            else:
                merged.append(dict(event))

        removed = len(note_events) - len(merged)
        if removed:
            print(f"  Smoothing: merged {removed} flicker(s)")

        return merged

    def annotate_pyin_agreement(self,
                                note_events: List[Dict[str, Any]],
                                pyin_data: List[Dict[str, Any]],
                                cent_tolerance: float = 50) -> List[Dict[str, Any]]:
        """
        Cross-validate each note event against PYIN pitch frames.

        For each note event, collects all PYIN frames whose timestamp falls
        within [start_time, end_time]. If any frame's frequency agrees with
        the note's frequency within cent_tolerance cents, sets pyin_agrees=True.
        Otherwise sets pyin_agrees=False and penalises confidence by ×0.85.

        When no PYIN frames fall in the window (absence of data), pyin_agrees
        is set to False but confidence is NOT penalised.
        """
        for event in note_events:
            note_freq = librosa.note_to_hz(event['note'])
            note_midi = librosa.hz_to_midi(note_freq)

            start = event['start_time']
            end   = event['end_time']

            window_frames = [
                f for f in pyin_data
                if start <= f['time'] <= end
            ]

            if not window_frames:
                event['pyin_agrees'] = False
                continue

            agrees = False
            for frame in window_frames:
                frame_midi = librosa.hz_to_midi(frame['frequency'])
                cents_diff = abs(note_midi - frame_midi) * 100
                if cents_diff <= cent_tolerance:
                    agrees = True
                    break

            event['pyin_agrees'] = agrees
            if not agrees:
                event['confidence'] = event['confidence'] * 0.85

        return note_events

    def annotate_chord_fit(
        self,
        note_events: List[Dict[str, Any]],
        chord_timeline: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Add a chord_fit bool field to each note event.

        For each note, finds the chord entry whose time is closest to the
        note's start_time, then checks whether the note's pitch class
        appears in that chord's chord_notes list.

        Informational only — does not remove notes.

        Args:
            note_events:    List of note event dicts
            chord_timeline: Output of get_chord_timeline() — each entry is
                            {'time': float, 'chord_notes': List[str]}

        Returns:
            The same list with chord_fit: bool added to every event
        """
        if not chord_timeline:
            for event in note_events:
                event['chord_fit'] = False
            return note_events

        beat_times = np.array([entry['time'] for entry in chord_timeline])

        def _pitch_class(note: str) -> str:
            return ''.join(ch for ch in note if not ch.isdigit()).strip()

        for event in note_events:
            t   = event['start_time']
            idx = int(np.argmin(np.abs(beat_times - t)))
            pc  = _pitch_class(event.get('note', ''))
            event['chord_fit'] = pc in chord_timeline[idx]['chord_notes']

        return note_events

    def process(self, pitch_data: List[Dict[str, float]],
                pyin_data: Optional[List[Dict[str, Any]]] = None,
                chord_timeline: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        filtered_data = self.filter_by_confidence(pitch_data)
        note_events = self.segment_notes(filtered_data)
        result = [event.to_dict() for event in note_events]

        if self.apply_key_filter:
            key_result = detect_key(result)
            result = self.filter_by_key(result, key_result)

        if self.smooth:
            result = self.remove_octave_outliers(result)
            result = self.merge_flickers(result)

        if pyin_data is not None:
            result = self.annotate_pyin_agreement(result, pyin_data)

        # Chord annotation before transposition — pitch classes must be in concert pitch
        if chord_timeline is not None:
            result = self.annotate_chord_fit(result, chord_timeline)

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
              transpose: int = 0,
              apply_key_filter: bool = True,
              smooth: bool = True,
              pyin_data: Optional[List[Dict[str, Any]]] = None,
              chord_timeline: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Convenience function for note mapping.

    Args:
        pitch_data:           Raw pitch detection output
        confidence_threshold: Minimum confidence threshold
        transpose:            Semitones to transpose (default 0, use 9 for alto sax Eb)
        apply_key_filter:     Remove notes outside the detected key (default True)
        smooth:               Merge flickers and remove octave outliers (default True)
        pyin_data:            Optional PYIN frames for cross-validation (Step 4)
        chord_timeline:       Optional beat-level chord timeline (Step 5)

    Returns:
        List of note event dictionaries
    """
    mapper = NoteMapper(confidence_threshold=confidence_threshold,
                        transpose=transpose,
                        apply_key_filter=apply_key_filter,
                        smooth=smooth)
    return mapper.process(pitch_data, pyin_data=pyin_data, chord_timeline=chord_timeline)
    return mapper.process(pitch_data, pyin_data=pyin_data)


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