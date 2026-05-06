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
    from note_mapper import map_notes, detect_key
    from lyric_aligner import align_lyrics_to_notes
    from output import format_output
    from beat_tracker import get_beat_grid, BeatTracker
    from config import load_config
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Make sure all pipeline modules are in the 'pipeline' directory.")
    sys.exit(1)


def main():
    """Main CLI entry point for Melody Transcriber."""

    # Load config file defaults (CLI flags always override these)
    _cfg = load_config()

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
        default=_cfg.get('confidence', 0.85),
        help="Confidence threshold for note detection (0-1, default: 0.85)"
    )

    parser.add_argument(
        "--transpose",
        type=int,
        default=_cfg.get('transpose', 9),
        help="Semitones to transpose output (default: 9 for alto sax Eb). Use 0 for concert pitch."
    )

    parser.add_argument(
        "--output",
        default=_cfg.get('output', None),
        help="Path to write output file (optional, prints to stdout if not specified)"
    )

    parser.add_argument(
        "--format",
        choices=["table", "csv", "json", "leadsheet"],
        default=_cfg.get('format', 'table'),
        help="Output format (default: table)"
    )
    parser.add_argument(
        '--dual-pitch', action='store_true',
        default=_cfg.get('dual_pitch', False),
        help='Run PYIN pitch detection for cross-validation (slower)'
    )
    parser.add_argument(
        '--chord-context', action='store_true',
        default=_cfg.get('chord_context', False),
        help='Extract chord timeline from backing track for cross-validation (slower)'
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
    if _cfg:
        print(f"Config:             meloscribe.toml loaded ({len(_cfg)} setting(s))")
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

        # Phase 2b: Detect beat grid (always runs — fast, no extra deps)
        print("[Phase 2b/5] Detecting beat grid...")
        try:
            beat_grid = get_beat_grid(vocals_path)
            print(f"  ✓ {beat_grid['bpm']:.1f} BPM, {len(beat_grid['beat_times'])} beats")
        except Exception as e:
            print(f"  ⚠ Beat detection failed ({e}), continuing without beat grid")
            beat_grid = None
        print()

        # Phase 2c: PYIN cross-validation (optional)
        if args.dual_pitch:
            print("[Phase 2c/5] Running PYIN pitch cross-validation...")
            try:
                from pyin_detector import get_pyin_pitch
                pyin_data = get_pyin_pitch(vocals_path)
                print(f"  ✓ {len(pyin_data)} voiced PYIN frames")
            except Exception as e:
                print(f"  ⚠ PYIN detection failed ({e}), continuing without")
                pyin_data = None
        else:
            pyin_data = None
        print()

        # Phase 2d: Chord context (optional)
        if args.chord_context:
            print("[Phase 2d/5] Extracting chord context from backing track...")
            try:
                from chord_tracker import get_chord_timeline
                from stemmer import get_stem_paths
                stem_paths = get_stem_paths(args.input or args.vocals)
                beat_times = beat_grid['beat_times'] if beat_grid else []
                chord_timeline = get_chord_timeline(stem_paths['other'], beat_times)
                print(f"  ✓ {len(chord_timeline)} chord entries")
            except Exception as e:
                print(f"  ⚠ Chord extraction failed ({e}), continuing without")
                chord_timeline = None
        else:
            chord_timeline = None
        print()

        # Phase 3: Map pitch to notes with filtering and transposition
        print("[Phase 3/5] Mapping pitch to notes...")
        note_events = map_notes(pitch_data, args.confidence,
                                transpose=args.transpose,
                                pyin_data=pyin_data,
                                chord_timeline=chord_timeline)
        print(f"  ✓ Mapped to {len(note_events)} note events")

        # Annotate beat alignment if beat grid is available
        if beat_grid:
            tracker = BeatTracker()
            note_events = tracker.annotate_beat_alignment(note_events, beat_grid)

        if pyin_data:
            agreed = sum(1 for e in note_events if e.get('pyin_agrees'))
            print(f"  PYIN agreement: {agreed}/{len(note_events)} notes confirmed")

        if chord_timeline:
            fit = sum(1 for e in note_events if e.get('chord_fit'))
            print(f"  Chord fit: {fit}/{len(note_events)} notes fit active chord")
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

        # Key detection
        key_result = detect_key(aligned_events)
        candidates = ', '.join(f"{k} ({v:.0%})" for k, v in key_result['candidates'])
        print(f"  • Estimated key:  {key_result['key']} ({key_result['score']:.0%} match)")
        print(f"  • Also possible:  {candidates}")

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
        print(f"  • Estimated key:  {key_result['key']} ({key_result['score']:.0%} match)")
        print(f"  • Also possible:  {candidates}") 
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