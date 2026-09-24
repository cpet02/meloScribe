"""Benchmark runner.

    python -m meloscribe.eval.runner --systems basic_pitch,pyin
    python -m meloscribe.eval.runner --systems oracle          # harness self-test
    python -m meloscribe.eval.runner --dataset data/vocadito --systems basic_pitch

Results are written to JSON so runs can be diffed: `--baseline` prints the
per-metric delta against an earlier run, which is the only honest way to claim
a change improved anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from .datasets import (AUDIO_EXTENSIONS, is_auxiliary, notes_path_for,
                       read_notes_csv)
from .groundtruth import GroundTruth, load_csv_f0
from .metrics import ScoreResult, aggregate, format_table, score
from .systems import REGISTRY, System, get_system

DEFAULT_SYNTH_DIR = Path('data/synthetic')
DEFAULT_RESULTS_DIR = Path('data/results')


def load_dataset(path) -> List[GroundTruth]:
    """Load a corpus laid out as matching audio/annotation pairs.

    Expects `<stem>.wav` (or .mp3/.flac/.ogg) beside `<stem>.csv` holding
    `time,frequency` rows, and optionally `<stem>.notes.csv` holding
    `onset,offset,midi` rows - see `datasets.py`, which also converts the
    public corpora (vocadito, iKala, TONAS, MedleyDB) into this layout.

    When reference notes exist they are what note F1 is scored against;
    otherwise notes are re-derived from the f0 curve, which grades the
    segmentation against a heuristic rather than against an annotator.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")

    truths: List[GroundTruth] = []
    for csv_path in sorted(path.rglob('*.csv')):
        if is_auxiliary(csv_path):
            continue  # x.notes.csv and the like annotate track x
        audio = None
        for ext in AUDIO_EXTENSIONS:
            candidate = csv_path.with_suffix(ext)
            if candidate.exists():
                audio = candidate
                break
        if audio is None:
            print(f"  skipping {csv_path.name}: no matching audio file")
            continue
        truth = load_csv_f0(csv_path, name=csv_path.stem, audio_path=audio)
        notes_path = notes_path_for(csv_path)
        if notes_path.exists():
            truth.notes = read_notes_csv(notes_path) or None
        truths.append(truth)

    if not truths:
        raise ValueError(f"No audio/CSV annotation pairs found under {path}")
    return truths


def run_system(system: System, truths: List[GroundTruth],
               verbose: bool = True) -> List[ScoreResult]:
    """Score one system over every track, surviving per-track failures.

    A crash on one track must not lose the results for the rest of the run;
    it is recorded as a zero-scoring result so the failure stays visible in
    the table rather than silently shrinking the track count.
    """
    results: List[ScoreResult] = []

    for truth in truths:
        if truth.audio_path is None or not truth.audio_path.exists():
            print(f"  {truth.name}: audio missing, skipped")
            continue
        try:
            prediction = system.run(truth.audio_path)
            result = score(truth, prediction, system=system.name)
        except Exception as exc:
            print(f"  {truth.name}: FAILED ({exc.__class__.__name__}: {exc})")
            if verbose:
                traceback.print_exc(limit=3)
            result = ScoreResult(track=truth.name, system=system.name,
                                 meta={'error': str(exc)})
        results.append(result)
        if verbose:
            print(f"  {truth.name:28s} OA={result.headline:.3f} "
                  f"({result.runtime_s:.1f}s)")

    return results


def save_results(results: Dict[str, List[ScoreResult]], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: {
            'aggregate': aggregate(rows),
            'tracks': [r.flat() for r in rows],
        }
        for name, rows in results.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"\nResults written to {path}")


def print_delta(current: Dict[str, List[ScoreResult]], baseline_path) -> None:
    """Print metric-by-metric change against a previous results file."""
    baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))

    print(f"\nDelta vs {baseline_path}")
    print("-" * 60)
    for name, rows in current.items():
        if name not in baseline:
            print(f"{name}: not present in baseline")
            continue
        now = aggregate(rows)
        before = baseline[name]['aggregate']
        print(f"\n{name}")
        for key in ('Overall Accuracy', 'Raw Pitch Accuracy',
                    'Raw Chroma Accuracy', 'octave_error_rate', 'note_f1',
                    'runtime_s'):
            if key not in now or key not in before:
                continue
            change = now[key] - before[key]
            arrow = '+' if change > 0 else ''
            print(f"  {key:22s} {before[key]:.4f} -> {now[key]:.4f} "
                  f"({arrow}{change:.4f})")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Score melody transcription systems against ground truth.')
    parser.add_argument('--systems', default='basic_pitch',
                        help='Comma-separated system names (default: basic_pitch)')
    parser.add_argument('--dataset',
                        help='Directory of audio + time,frequency CSV pairs. '
                             'Defaults to the built-in synthetic set.')
    parser.add_argument('--synth-dir', default=str(DEFAULT_SYNTH_DIR),
                        help='Where synthetic benchmark audio is rendered')
    parser.add_argument('--rebuild-synth', action='store_true',
                        help='Re-render synthetic audio even if it exists')
    parser.add_argument('--tracks', help='Comma-separated track names to limit the run to')
    parser.add_argument('--output', help='Write results JSON here')
    parser.add_argument('--baseline', help='Compare against a previous results JSON')
    parser.add_argument('--list', action='store_true', help='List systems and exit')
    args = parser.parse_args(argv)

    if args.list:
        for name, cls in sorted(REGISTRY.items()):
            print(f"  {name:14s} {getattr(cls, 'description', '')}")
        return 0

    if args.dataset:
        print(f"Loading dataset: {args.dataset}")
        truths = load_dataset(args.dataset)
    else:
        from .synth import build_dataset
        print(f"Building synthetic benchmark in {args.synth_dir}")
        truths = build_dataset(args.synth_dir, force=args.rebuild_synth)

    if args.tracks:
        wanted = {t.strip() for t in args.tracks.split(',')}
        truths = [t for t in truths if t.name in wanted]
        if not truths:
            print(f"No tracks matched {sorted(wanted)}", file=sys.stderr)
            return 1

    print(f"{len(truths)} track(s) loaded\n")

    results: Dict[str, List[ScoreResult]] = {}
    truth_index = {t.name: t for t in truths}

    for name in [s.strip() for s in args.systems.split(',') if s.strip()]:
        print(f"=== {name} ===")
        try:
            # The oracle needs the answers handed to it; nothing else does.
            system = (get_system(name, truths=truth_index)
                      if name == 'oracle' else get_system(name))
        except Exception as exc:
            print(f"  cannot construct system: {exc}", file=sys.stderr)
            continue

        rows = run_system(system, truths)
        results[name] = rows
        print()
        print(format_table(rows))
        print()

    if not results:
        print("No systems ran.", file=sys.stderr)
        return 1

    if args.output:
        save_results(results, args.output)
    if args.baseline:
        print_delta(results, args.baseline)

    return 0


if __name__ == '__main__':
    sys.exit(main())
