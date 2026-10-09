"""Compare saved raw-visit generations on a fixed, recorded set of positions."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/user/chess/alphazero_chess_v2")
import chess
from flax import linen as nn, serialization
import jax
import jax.numpy as jnp
import numpy as np

from azchess.checkpoint import load_variables
from azchess.game import encode_board

PROJECT = Path("/home/user/chess/alphazero_chess_v2")
OUTPUT = Path(__file__).parent / "value_generation_trace"
SAVED = PROJECT / "experiments/raw-visits-v1/checkpoints/saved"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, nargs="+", default=[0, 25, 100, 200, 400])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((PROJECT / "experiments/raw-visits-v1/selfplay_data").glob("*.npz"))
    metadata = []
    for path in files:
        with np.load(path) as data:
            labels = data["values"]
            metadata.append({"path": str(path), "mtime": path.stat().st_mtime,
                             "termination": str(data["termination"]),
                             "labels": {str(v): int((labels == v).sum()) for v in (0, 1, -1)}})

    manifest_path = output / "positions.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        rng = np.random.default_rng(20260928)
        manifest = [{"moves": moves, "group": "opening"} for moves in
                    [[], ["e2e4", "e7e5"], ["d2d4", "d7d5", "c2c4"]]]
        for termination in ("THREEFOLD_REPETITION", "CHECKMATE"):
            candidates = [item for item in metadata[-1000:] if item["termination"] == termination]
            selected = rng.choice(len(candidates), min(8, len(candidates)), replace=False)
            for index in selected:
                path = Path(candidates[index]["path"])
                with np.load(path) as data:
                    for label in ([0] if termination == "THREEFOLD_REPETITION" else [1, -1]):
                        positions = np.flatnonzero(data["values"] == label)
                        chosen = rng.choice(positions, min(8 if label == 0 else 4, len(positions)), replace=False)
                        manifest.extend({"file": str(path), "position": int(p), "group": str(label)} for p in chosen)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    states = []
    for item in manifest:
        if "moves" in item:
            board = chess.Board()
            for move in item["moves"]:
                board.push_uci(move)
            states.append(encode_board(board))
        else:
            with np.load(item["file"]) as data:
                states.append(data["states"][item["position"]].astype(np.float32))
    inputs = jnp.asarray(np.stack(states))
    model, template = load_variables(SAVED / "base-cycle-000119.msgpack")

    @jax.jit
    def inspect(variables, inputs):
        (_, value), extra = model.apply(
            variables, inputs, train=False,
            capture_intermediates=lambda module, method: isinstance(module, nn.Dense),
            mutable=["intermediates"],
        )
        return value, extra["intermediates"]["Dense_0"]["__call__"][0]

    results_path = output / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    groups = np.asarray([item["group"] for item in manifest])
    for cycle in args.cycles:
        key = str(cycle)
        if key in results:
            print("CACHED", key, flush=True)
            continue
        path = SAVED / ("base-cycle-000119.msgpack" if cycle == 0 else f"cycle-{cycle:06d}.msgpack")
        payload = path.read_bytes()
        variables = serialization.from_bytes(template, payload)
        values, dense = inspect(variables, inputs)
        values, dense = np.asarray(values), np.asarray(dense)
        dead = np.all(dense <= 0, axis=1)
        fixed = float(np.tanh(np.asarray(variables["params"]["Dense_1"]["bias"])[0]))
        group_stats = {}
        for group in ("opening", "0", "1", "-1"):
            mask = groups == group
            sample = values[mask]
            group_stats[group] = {"positions": len(sample), "dead_positions": int(dead[mask].sum()),
                                  "mean": float(sample.mean()), "std": float(sample.std()),
                                  "min": float(sample.min()), "max": float(sample.max())}
        before = [item for item in metadata if item["mtime"] <= path.stat().st_mtime]
        recent = before[-1000:]
        outcomes = Counter(item["termination"] for item in recent)
        label_counts = Counter()
        for item in recent:
            label_counts.update(item["labels"])
        result = {"checkpoint": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                  "mtime": path.stat().st_mtime, "fixed_value": fixed,
                  "groups": group_stats, "data_games_before_checkpoint": len(before),
                  "recent_data_games": len(recent), "recent_outcomes": dict(outcomes),
                  "recent_label_counts": dict(label_counts)}
        results[key] = result
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("GENERATION", cycle, json.dumps({
            "dead_positions": int(dead.sum()), "positions": len(values),
            "opening_dead": group_stats["opening"]["dead_positions"],
            "draw_dead": group_stats["0"]["dead_positions"],
            "fixed_value": fixed, "recent_outcomes": dict(outcomes),
        }), flush=True)


if __name__ == "__main__":
    main()
