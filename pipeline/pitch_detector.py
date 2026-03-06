"""
Phase 2: Pitch Detection using basic-pitch
Extracts pitch contours from audio files with time, frequency, and confidence.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import warnings
import sys
from pathlib import Path

try:
    from basic_pitch.inference import predict as basic_pitch_predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False
    warnings.warn("basic-pitch not installed. Please install with: pip install basic-pitch")


def _midi_to_hz(midi_pitch: float) -> float:
    """Convert MIDI pitch number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))


class PitchDetector:
    """Wrapper for basic-pitch pitch detection."""

    def __init__(self, onset_threshold: float = 0.5,
                 frame_threshold: float = 0.3,
                 minimum_note_length: float = 0.058):
        """
        Initialize the pitch detector.

        Args:
            onset_threshold: Onset detection threshold (0-1)
            frame_threshold: Frame-level activation threshold (0-1)
            minimum_note_length: Minimum note length in seconds
        """
        if not BASIC_PITCH_AVAILABLE:
            raise ImportError(
                "basic-pitch is not installed. Please install with: pip install basic-pitch"
            )

        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_length = minimum_note_length

    def detect_pitch(self, audio_path: str) -> List[Dict[str, float]]:
        """
        Detect pitch from an audio file using basic-pitch.

        Args:
            audio_path: Path to input WAV file

        Returns:
            List of dicts with keys: 'time', 'frequency', 'confidence'
            - time: start time of the note in seconds
            - frequency: frequency in Hz
            - confidence: amplitude/confidence value (0-1)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is not a WAV file
            RuntimeError: If pitch detection fails
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if audio_path.suffix.lower() != '.wav':
            raise ValueError(f"Audio file must be WAV format. Got: {audio_path.suffix}")

        print(f"Detecting pitch from: {audio_path.name}")

        try:
            # basic_pitch_predict returns (model_output, midi_data, note_events)
            # note_events is a list of tuples:
            # (start_time_s, end_time_s, pitch_midi, amplitude, pitch_bends)
            _, _, note_events = basic_pitch_predict(
                audio_path=str(audio_path),
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
                minimum_note_length=self.minimum_note_length,
                multiple_pitch_bends=False,
                melodia_trick=True,
            )

            if not note_events:
                print("Warning: No pitch events detected")
                return []

            pitch_data = []
            for start_time, end_time, midi_pitch, amplitude, _ in note_events:
                frequency = _midi_to_hz(midi_pitch)

                pitch_data.append({
                    'time': float(start_time),
                    'frequency': float(frequency),
                    'confidence': float(amplitude)
                })

            # Sort by time
            pitch_data.sort(key=lambda x: x['time'])

            print(f"Detected {len(pitch_data)} pitch events")
            return pitch_data

        except Exception as e:
            raise RuntimeError(f"Pitch detection failed: {e}")


def detect_pitch(audio_path: str, **kwargs) -> List[Dict[str, float]]:
    """
    Convenience function for pitch detection.

    Args:
        audio_path: Path to input WAV file
        **kwargs: Additional arguments passed to PitchDetector

    Returns:
        List of dicts with keys: 'time', 'frequency', 'confidence'
    """
    detector = PitchDetector(**kwargs)
    return detector.detect_pitch(audio_path)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detect pitch from WAV file using basic-pitch")
    parser.add_argument("input", help="Path to input WAV file")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--frame-threshold", type=float, default=0.3)

    args = parser.parse_args()

    try:
        pitch_data = detect_pitch(
            args.input,
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(pitch_data, f, indent=2)
            print(f"Saved {len(pitch_data)} pitch events to {args.output}")
        else:
            print(f"\nDetected {len(pitch_data)} pitch events")
            if pitch_data:
                print(f"Time range: {pitch_data[0]['time']:.3f}s to {pitch_data[-1]['time']:.3f}s")
                print(f"Frequency range: {min(d['frequency'] for d in pitch_data):.1f}Hz to "
                      f"{max(d['frequency'] for d in pitch_data):.1f}Hz")
                print(f"Confidence range: {min(d['confidence'] for d in pitch_data):.3f} to "
                      f"{max(d['confidence'] for d in pitch_data):.3f}")
                print("\nFirst 5 events:")
                for i, event in enumerate(pitch_data[:5]):
                    print(f"  {i}: t={event['time']:.3f}s, "
                          f"f={event['frequency']:.1f}Hz, "
                          f"c={event['confidence']:.3f}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)