"""Voter ablation.

Scores every subset of the voter set so a gain or loss can be attributed to a
specific voter rather than to "the ensemble". A voter that does not improve
the numbers is costing runtime for nothing and should be dropped or reweighted;
without this, an ensemble accumulates dead weight indefinitely.

    python -m meloscribe.eval.ablate --voters pyin,basic_pitch,harmonic_template
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from typing import List, Optional

from .metrics import aggregate, score
from .synth import build_dataset
from .systems import EnsembleSystem


def run(voter_names: List[str], synth_dir: str, tracks: Optional[List[str]],
        max_size: Optional[int], singles_only: bool) -> int:
    warnings.filterwarnings('ignore')

    truths = build_dataset(synth_dir)
    if tracks:
        truths = [t for t in truths if t.name in set(tracks)]
    if not truths:
        print('No tracks matched.', file=sys.stderr)
        return 1

    sizes = [1] if singles_only else range(1, (max_size or len(voter_names)) + 1)
    combos = [c for size in sizes
              for c in itertools.combinations(voter_names, size)]

    print(f"{len(combos)} voter combination(s) over {len(truths)} track(s)\n")
    header = f"{'voters':52s} {'OA':>6s} {'RPA':>6s} {'RCA':>6s} {'OCT':>6s} {'noteF1':>7s} {'sec':>6s}"
    print(header)
    print('-' * len(header))

    rows = []
    for combo in combos:
        system = EnsembleSystem(voters=combo)
        results = []
        for truth in truths:
            try:
                results.append(score(truth, system.run(truth.audio_path),
                                     system='+'.join(combo)))
            except Exception as exc:
                print(f"  {'+'.join(combo)} failed on {truth.name}: {exc}",
                      file=sys.stderr)
        if not results:
            continue

        agg = aggregate(results)
        rows.append((agg.get('Overall Accuracy', 0), combo, agg))
        print(f"{'+'.join(combo):52s} "
              f"{agg.get('Overall Accuracy', 0):6.3f} "
              f"{agg.get('Raw Pitch Accuracy', 0):6.3f} "
              f"{agg.get('Raw Chroma Accuracy', 0):6.3f} "
              f"{agg.get('octave_error_rate', 0):6.3f} "
              f"{agg.get('note_f1', 0):7.3f} "
              f"{agg.get('runtime_s', 0):6.1f}")

    if rows:
        rows.sort(key=lambda r: -r[0])
        print(f"\nBest by OA: {'+'.join(rows[0][1])} ({rows[0][0]:.3f})")
        by_f1 = max(rows, key=lambda r: r[2].get('note_f1', 0))
        print(f"Best by note F1: {'+'.join(by_f1[1])} "
              f"({by_f1[2].get('note_f1', 0):.3f})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voters',
                        default='crepe,basic_pitch,pyin,harmonic_template',
                        help='Comma-separated voters to permute')
    parser.add_argument('--synth-dir', default='data/synthetic')
    parser.add_argument('--tracks', help='Comma-separated track names')
    parser.add_argument('--max-size', type=int,
                        help='Largest subset size to test')
    parser.add_argument('--singles-only', action='store_true',
                        help='Only score each voter alone')
    args = parser.parse_args(argv)

    return run([v.strip() for v in args.voters.split(',') if v.strip()],
               args.synth_dir,
               [t.strip() for t in args.tracks.split(',')] if args.tracks else None,
               args.max_size, args.singles_only)


if __name__ == '__main__':
    sys.exit(main())
