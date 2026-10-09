"""Check that an isolated self-play directory contains raw MCTS visit targets."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--min-games", type=int, default=1)
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.npz"))
    if len(files) < args.min_games:
        raise SystemExit(f"Expected at least {args.min_games} games; found {len(files)}")

    outcomes: Counter[str] = Counter()
    positions = 0
    for path in files:
        with np.load(path) as game:
            if "policy_target_kind" not in game or str(game["policy_target_kind"]) != "raw_mcts_visits_v1":
                raise ValueError(f"Wrong policy target kind: {path}")
            moves = game["moves"]
            offsets = game["policy_offsets"]
            values = game["policy_values"].astype(np.float32)
            if len(moves) != len(game["states"]) or len(offsets) != len(moves) + 1:
                raise ValueError(f"Position count mismatch: {path}")
            if offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) <= 0):
                raise ValueError(f"Invalid policy offsets: {path}")
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid policy values: {path}")
            sums = np.add.reduceat(values, offsets[:-1])
            if not np.allclose(sums, 1.0, atol=0.01):
                raise ValueError(f"Policy targets do not sum to one: {path}")
            outcomes[str(game["termination"])] += 1
            positions += len(moves)
            print(f"OK {path.name}: {len(moves)} plies, {game['termination']}", flush=True)

    print(f"VERIFIED {len(files)} games, {positions} positions, outcomes={dict(outcomes)}", flush=True)


if __name__ == "__main__":
    main()
