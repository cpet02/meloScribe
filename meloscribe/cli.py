"""Command-line interface.

    meloscribe song.mp3 --track "Karma Police" --artist Radiohead
    meloscribe song.mp3 --no-lyrics --format leadsheet
    meloscribe vocals.wav --vocals-only --midi out.mid
    meloscribe song.mp3 --no-lyrics --transpose 9 --musicxml sax.musicxml

The naming gate is enforced here as it is everywhere else: a track name is
required unless `--no-lyrics` is passed. The error arrives before separation
starts, not after.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import List, Optional

from .audio import AudioLoadError
from .lyrics.service import LyricsMode, MissingTrackName
from .output import (FORMATS, musicxml_for_output, render, score_for_output,
                     write_midi)
from .pipeline import Pipeline, TranscriptionRequest
from .stems import best_device


def _progress_bar(width: int = 34):
    """A single-line progress bar that rewrites itself in place."""
    state = {'last': ''}

    def report(fraction: float, message: str) -> None:
        filled = int(width * max(0.0, min(1.0, fraction)))
        bar = '#' * filled + '.' * (width - filled)
        line = f"\r[{bar}] {fraction * 100:3.0f}%  {message[:38]:38s}"
        if line != state['last']:
            sys.stderr.write(line)
            sys.stderr.flush()
            state['last'] = line

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='meloscribe',
        description='Transcribe a vocal melody from an audio file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  meloscribe song.mp3 --track "Karma Police" --artist Radiohead
  meloscribe song.mp3 --no-lyrics --format leadsheet
  meloscribe song.mp3 --preset max --transpose 9 --midi sax.mid
  meloscribe song.mp3 --transpose 9 --musicxml sax.musicxml --pickup 1
  meloscribe vocals.wav --vocals-only --format csv -o notes.csv
""")

    parser.add_argument('input', help='Audio file (mp3, wav, flac, m4a ...)')
    parser.add_argument('--track', '-t', default='',
                        help='Track name. Required unless --no-lyrics.')
    parser.add_argument('--artist', '-a', default='',
                        help='Artist name, to disambiguate the lyric lookup')

    lyrics = parser.add_argument_group('lyrics')
    lyrics.add_argument('--no-lyrics', action='store_true',
                        help='Skip lyrics entirely (bypasses the name requirement)')
    lyrics.add_argument('--lyrics-mode', choices=[m.value for m in LyricsMode],
                        default=LyricsMode.ALIGN.value,
                        help='off | lookup | align (default) | transcribe')

    audio = parser.add_argument_group('audio')
    audio.add_argument('--preset', choices=['fast', 'balanced', 'max'],
                       default='balanced', help='Separation quality/time trade-off')
    audio.add_argument('--vocals-only', action='store_true',
                       help='Input is already an isolated vocal; skip separation')
    audio.add_argument('--device', help='torch device (cuda, cpu). Auto-detected.')
    audio.add_argument('--voters', help='Comma-separated pitch voters to use')
    audio.add_argument('--force', action='store_true',
                       help='Ignore cached stems and re-separate')
    audio.add_argument('--rhythm', action='store_true',
                       help='Also track the beat and score how metrically '
                            'plausible the transcription is (diagnostic)')

    out = parser.add_argument_group('output')
    out.add_argument('--format', '-f', choices=FORMATS, default='table')
    out.add_argument('--output', '-o', help='Write to this file instead of stdout')
    out.add_argument('--midi', help='Also write a MIDI file here')
    out.add_argument('--musicxml', help='Also write sheet music (MusicXML) '
                                        'here, for MuseScore and the like')
    out.add_argument('--quantize', action='store_true',
                     help='Write --midi on the same beat grid as the sheet '
                          'music, instead of the performed timing')
    out.add_argument('--time-signature', choices=['4/4', '3/4'],
                     default='4/4',
                     help='Meter for sheet music; not detected (default 4/4)')
    out.add_argument('--pickup', type=int, default=0, metavar='BEATS',
                     help='Beats before the first bar line in sheet music, '
                          'counted from the beat the melody starts on; not '
                          'detected (default 0)')
    out.add_argument('--transpose', type=int, default=0,
                     help='Semitones to transpose (9 for alto sax in Eb)')
    out.add_argument('--confidence', type=float, default=0.0,
                     help='Drop notes below this confidence (0-1, default: keep all)')
    out.add_argument('--show-voters', action='store_true',
                     help='Include the per-voter breakdown in table output')
    out.add_argument('--quiet', '-q', action='store_true',
                     help='Suppress progress output')

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    mode = LyricsMode.OFF if args.no_lyrics else LyricsMode(args.lyrics_mode)

    request = TranscriptionRequest(
        input_path=Path(args.input),
        track_name=args.track,
        artist_name=args.artist,
        lyrics_mode=mode,
        separation_preset=args.preset,
        voters=[v.strip() for v in args.voters.split(',')] if args.voters else None,
        transpose=args.transpose,
        confidence_threshold=args.confidence,
        device=args.device,
        assess_rhythm=args.rhythm,
        force=args.force,
        vocals_only=args.vocals_only,
    )

    progress = None if args.quiet else _progress_bar()

    if not args.quiet:
        print(f"device: {best_device(args.device)}  preset: {args.preset}  "
              f"lyrics: {mode.value}", file=sys.stderr)

    try:
        # stdout is reserved for the rendered result. Libraries print progress
        # there (basic-pitch: "Predicting MIDI for ..."), which made
        # `--format json > notes.json` write invalid JSON.
        with contextlib.redirect_stdout(sys.stderr):
            output = Pipeline().run(request, progress=progress)
    except MissingTrackName as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, AudioLoadError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\ninterrupted', file=sys.stderr)
        return 130

    if not args.quiet:
        sys.stderr.write('\n')
        if output.key:
            line = f"key: {output.key.name} (confidence {output.key.confidence:.2f})"
            # Transposed notes are read in a different key from the one sung;
            # naming only the concert key would contradict every note.
            if output.written_key:
                line += f"  written: {output.written_key.name} ({args.transpose:+d})"
            print(line, file=sys.stderr)
        if output.lyrics:
            print(f"lyrics: {output.lyrics.summary()}", file=sys.stderr)
        if output.rhythm:
            print(output.rhythm.summary(), file=sys.stderr)
        print(f"notes: {len(output.notes)}  "
              f"mean confidence: {output.mean_confidence:.2f}  "
              f"elapsed: {output.elapsed_s:.1f}s", file=sys.stderr)
        for warning in output.warnings:
            print(f"warning: {warning}", file=sys.stderr)

    # Sheet music and quantised MIDI share one quantisation, so they agree.
    # Tracking the beat for it can take a few seconds, so only when asked.
    score = None
    if args.quantize and not args.midi:
        print('warning: --quantize applies to --midi; no MIDI requested',
              file=sys.stderr)
    if args.format == 'musicxml' or args.musicxml or (args.midi
                                                      and args.quantize):
        score = score_for_output(
            output, beats_per_bar=int(args.time_signature.split('/')[0]),
            pickup=args.pickup)
        if not args.quiet:
            print(_describe_score(score), file=sys.stderr)
            for warning in score.warnings:
                print(f"warning: {warning}", file=sys.stderr)

    if args.midi:
        try:
            write_midi(output.notes, args.midi,
                       score=score if args.quantize else None)
            if not args.quiet:
                print(f"midi: {args.midi}", file=sys.stderr)
        except ImportError as exc:
            print(f"warning: {exc}", file=sys.stderr)

    if args.musicxml:
        Path(args.musicxml).write_text(musicxml_for_output(output, score),
                                       encoding='utf-8')
        if not args.quiet:
            print(f"musicxml: {args.musicxml}", file=sys.stderr)

    if args.format == 'midi':
        if not args.midi:
            print('error: --format midi requires --midi PATH', file=sys.stderr)
            return 1
        return 0

    extra = {}
    if output.key and args.format == 'json':
        extra['key'] = output.key.name
        if output.written_key:
            extra['written_key'] = output.written_key.name
    if args.format == 'musicxml':
        text = musicxml_for_output(output, score)
    else:
        text = render(output.notes, args.format, show_voters=args.show_voters,
                      title=args.track, artist=args.artist, **extra)

    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
        if not args.quiet:
            print(f"written: {args.output}", file=sys.stderr)
    else:
        print(text)

    return 0


def _describe_score(score) -> str:
    """One line on how the sheet music was barred, and how far to trust it."""
    meter = f"{score.beats_per_bar}/4"
    if score.approximate:
        return (f"sheet music: {meter} at ~{score.bpm:.0f} BPM - approximate, "
                f"no reliable beat grid (tempo from note spacing)")
    triplets = len(score.ternary_beats)
    return (f"sheet music: {meter} at {score.bpm:.0f} BPM on the tracked beat "
            f"grid" + (f", {triplets} beat(s) in triplets" if triplets else ''))


if __name__ == '__main__':
    sys.exit(main())
