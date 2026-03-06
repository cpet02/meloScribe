#!/usr/bin/env python3
"""
Melody Transcriber - Main CLI Entry Point
Wires together all pipeline modules into a complete transcription workflow.
"""

import argparse
import sys
import os
from pathlib import Path

# Add pipeline directory to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

# Import pipeline modules
try:
    from stemmer import isolate_vocals
    from pitch_detector import detect_pitch
    from note_mapper import map_notes
    from lyric_aligner import align_lyrics_to_notes
    from output import format_output
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Make sure all pipeline modules are in the 'pipeline' directory.")
    sys.exit(1)


def main():
    """Main CLI entry point for Melody Transcriber."""
    parser = argparse.ArgumentParser(
        description="Melody Transcriber - Transcribe vocal melodies from MP3 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input song.mp3
  %(prog)s --input song.mp3 --lyrics song.lrc --format leadsheet --output notes.txt
  %(prog)s --input song.mp3 --confidence 0.9 --output notes.csv --format csv
  %(prog)s --vocals vocals.wav --confidence 0.75 --transpose 9
        """
    )

    # Input arguments (one of --input or --vocals required)
    parser.add_argument(
        "--input",
        required=False,
        help="Path to input MP3 file"
    )

    parser.add_argument(
        "--vocals",
        help="Path to pre-stemmed vocals.wav, skips Phase 1 stemming"
    )

    # Optional arguments
    parser.add_argument(
        "--lyrics",
        help="Path to LRC lyrics file (optional)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.85,
        help="Confidence threshold for note detection (0-1, default: 0.85)"
    )

    parser.add_argument(
        "--transpose",
        type=int,
        default=9,
        help="Semitones to transpose output (default: 9 for alto sax Eb). Use 0 for concert pitch."
    )

    parser.add_argument(
        "--output",
        help="Path to write output file (optional, prints to stdout if not specified)"
    )

    parser.add_argument(
        "--format",
        choices=["table", "csv", "json", "leadsheet"],
        default="table",
        help="Output format (default: table)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate that at least one input source is provided
    if not args.input and not args.vocals:
        parser.error("Either --input or --vocals must be provided.")

    if args.vocals and not os.path.exists(args.vocals):
        print(f"Error: Vocals file not found: {args.vocals}")
        sys.exit(1)

    if args.input and not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.lyrics and not os.path.exists(args.lyrics):
        print(f"Error: Lyrics file not found: {args.lyrics}")
        sys.exit(1)

    if not 0 <= args.confidence <= 1:
        print(f"Error: Confidence must be between 0 and 1, got {args.confidence}")
        sys.exit(1)

    if args.input and not args.input.lower().endswith('.mp3'):
        print(f"Warning: Input file '{args.input}' may not be an MP3 file.")

    print("=" * 60)
    print("Melody Transcriber - Starting Transcription")
    print("=" * 60)
    if args.input:
        print(f"Input:              {args.input}")
    if args.vocals:
        print(f"Vocals (pre-stemmed): {args.vocals}")
    if args.lyrics:
        print(f"Lyrics:             {args.lyrics}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Transpose:          {args.transpose} semitones", end="")
    print(" (alto sax Eb)" if args.transpose == 9 else " (concert pitch)" if args.transpose == 0 else "")
    print(f"Output format:      {args.format}")
    if args.output:
        print(f"Output file:        {args.output}")
    print()

    try:
        # Phase 1: Stem vocals from MP3 (or skip if provided)
        if args.vocals:
            print("[Phase 1/5] Using provided vocals file, skipping stemming...")
            vocals_path = args.vocals
            print(f"  ✓ Using: {vocals_path}")
        else:
            print("[Phase 1/5] Separating vocals from audio...")
            vocals_path = isolate_vocals(args.input)
            print(f"  ✓ Vocals isolated: {vocals_path}")
        print()

        # Phase 2: Detect pitch from vocals
        print("[Phase 2/5] Detecting pitch contours...")
        pitch_data = detect_pitch(vocals_path)
        print(f"  ✓ Detected {len(pitch_data)} pitch frames")
        print()

        # Phase 3: Map pitch to notes with filtering and transposition
        print("[Phase 3/5] Mapping pitch to notes...")
        note_events = map_notes(pitch_data, args.confidence, transpose=args.transpose)
        print(f"  ✓ Mapped to {len(note_events)} note events")
        print()

        # Phase 4: Align lyrics (if provided)
        print("[Phase 4/5] Aligning lyrics...")
        if args.lyrics:
            aligned_events = align_lyrics_to_notes(note_events, args.lyrics)
            matched_lyrics = sum(1 for event in aligned_events if event.get('lyric') is not None)
            print(f"  ✓ Aligned lyrics to {matched_lyrics} of {len(aligned_events)} notes")
        else:
            aligned_events = align_lyrics_to_notes(note_events, None)
            print(f"  ✓ No lyrics provided, all lyrics set to null")
        print()

        # Phase 5: Format and output results
        print("[Phase 5/5] Formatting output...")
        result = format_output(
            aligned_events,
            format_type=args.format,
            output_file=args.output
        )

        # Print to stdout if no output file specified
        if not args.output and result:
            print()
            print("=" * 60)
            print("Transcription Results")
            print("=" * 60)
            print(result)

        print()
        print("=" * 60)
        print("Transcription complete!")
        print("=" * 60)

        # Summary
        input_label = os.path.basename(args.vocals or args.input)
        print(f"Summary:")
        print(f"  • Input:       {input_label}")
        print(f"  • Note events: {len(aligned_events)}")
        print(f"  • Transposed:  {args.transpose} semitones")
        if args.lyrics:
            matched = sum(1 for event in aligned_events if event.get('lyric') is not None)
            print(f"  • Lyrics matched: {matched}/{len(aligned_events)}")
        if args.output:
            print(f"  • Output saved to: {args.output}")

    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: Invalid value - {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("Please ensure all required packages are installed:")
        print("  pip install demucs basic-pitch librosa soundfile numpy pandas tabulate")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTranscription cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during transcription: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()