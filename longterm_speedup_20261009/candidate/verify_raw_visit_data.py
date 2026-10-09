"""Check that an isolated self-play directory contains raw MCTS visit targets."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from azchess.data_cache import DataCache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--min-games", type=int, default=1)
    parser.add_argument("--full",action="store_true",help="Revalidate every file, bypassing metadata cache")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.npz"))
    if len(files) < args.min_games:
        raise SystemExit(f"Expected at least {args.min_games} games; found {len(files)}")

    outcomes: Counter[str] = Counter()
    positions = 0
    cache=DataCache(args.data_dir)
    cache.data['validation_runs']+=1
    full=args.full or cache.data['validation_runs']%20==0
    for path in files:
        info=cache.get(path,validate=True,full=full)
        outcomes[info['termination']]+=1;positions+=info['positions']
    cache.flush(files)
    print(f"Validation: {cache.misses} files checked, {cache.hits} unchanged cached; full={full}",flush=True)

    print(f"VERIFIED {len(files)} games, {positions} positions, outcomes={dict(outcomes)}", flush=True)


if __name__ == "__main__":
    main()
