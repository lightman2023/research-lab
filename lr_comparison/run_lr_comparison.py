"""Isolated paired learning-rate experiment using existing self-play only."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
PROJECT = Path("/home/user/chess/alphazero_chess_v2")
sys.path.insert(0, str(PROJECT))

import chess
from flax import linen as nn, serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from azchess.checkpoint import load_variables, save_bytes_atomic, save_variables
from azchess.game import ACTION_SIZE, encode_board
from train import ReplayBuffer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1:
        parser.error("updates and batch-size must be positive")
    root = args.output_dir or PROJECT / "experiments" / ("lr-cycle50-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, root / "run_lr_comparison.py")

    def emit(payload):
        line = json.dumps(payload, ensure_ascii=False)
        print(line, flush=True)
        with (root / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    source = PROJECT / "experiments/raw-visits-v1/checkpoints/saved/cycle-000050.msgpack"
    source_hash = digest(source)
    shutil.copy2(source, root / "base-cycle-000050.msgpack")
    eval_manifest = json.loads(args.eval_manifest.read_text(encoding="utf-8"))
    eval_sources = {item["file"] for item in eval_manifest if "file" in item}
    source_data = PROJECT / "experiments/raw-visits-v1/selfplay_data"
    candidates = sorted(path for path in source_data.glob("*.npz") if path.stat().st_mtime <= source.stat().st_mtime)
    candidates = candidates[-1000:]
    assert not (eval_sources & {str(path) for path in candidates}), "Evaluation games overlap training games"
    data_dir = root / "selfplay_data"
    data_dir.mkdir()
    training_manifest = []
    training_labels = {str(label): 0 for label in (0, 1, -1)}
    for path in candidates:
        destination = data_dir / path.name
        shutil.copy2(path, destination)
        training_manifest.append({"source": str(path), "copy": str(destination), "sha256": digest(destination)})
        with np.load(destination) as game:
            assert str(game["policy_target_kind"]) == "raw_mcts_visits_v1"
            for label in training_labels:
                training_labels[label] += int((game["values"] == int(label)).sum())
    (root / "training_manifest.json").write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")
    (root / "evaluation_manifest.json").write_text(json.dumps(eval_manifest, indent=2), encoding="utf-8")
    eval_hashes = {path: digest(Path(path)) for path in sorted(eval_sources)}
    (root / "evaluation_file_hashes.json").write_text(json.dumps(eval_hashes, indent=2), encoding="utf-8")

    eval_states, eval_policies, eval_labels, groups = [], [], [], []
    for item in eval_manifest:
        policy = np.zeros(ACTION_SIZE, np.float32)
        if "moves" in item:
            board = chess.Board()
            for move in item["moves"]:
                board.push_uci(move)
            state, label = encode_board(board), 0.0
        else:
            with np.load(item["file"]) as game:
                pos = item["position"]
                state, label = game["states"][pos].astype(np.float32), float(game["values"][pos])
                start, end = game["policy_offsets"][pos:pos + 2]
                policy[game["policy_indices"][start:end].astype(np.int32)] = game["policy_values"][start:end]
        eval_states.append(state)
        eval_policies.append(policy)
        eval_labels.append(label)
        groups.append(item["group"])
    x_eval = jnp.asarray(np.stack(eval_states))
    p_eval = jnp.asarray(np.stack(eval_policies))
    labels = np.asarray(eval_labels)
    groups = np.asarray(groups)
    model, base = load_variables(root / "base-cycle-000050.msgpack")

    @jax.jit
    def inspect(variables):
        (logits, value), extra = model.apply(
            variables, x_eval, train=False,
            capture_intermediates=lambda module, method: isinstance(module, nn.Dense),
            mutable=["intermediates"],
        )
        policy_loss = -jnp.sum(p_eval * jax.nn.log_softmax(logits), axis=1)
        return value, extra["intermediates"]["Dense_0"]["__call__"][0], policy_loss

    def evaluate(variables):
        values, dense, policy_losses = map(np.asarray, inspect(variables))
        assert np.isfinite(values).all() and np.isfinite(dense).all()
        dead = np.all(dense <= 0, axis=1)
        stats = {}
        for group in ("opening", "0", "1", "-1"):
            mask = groups == group
            stats[group] = {"positions": int(mask.sum()), "dead": int(dead[mask].sum()),
                            "value_mean": float(values[mask].mean()), "value_std": float(values[mask].std())}
            if group != "opening":
                stats[group]["value_mse"] = float(np.mean((values[mask] - labels[mask]) ** 2))
                stats[group]["policy_cross_entropy"] = float(policy_losses[mask].mean())
        data = groups != "opening"
        return {"dead": int(dead.sum()), "positions": len(values), "groups": stats,
                "value_mse": float(np.mean((values[data] - labels[data]) ** 2)),
                "policy_cross_entropy": float(policy_losses[data].mean())}

    @jax.jit
    def update(params, batch_stats, opt_state, x, target_policy, target_value, learning_rate):
        def loss_fn(current_params):
            (logits, value), updated = model.apply(
                {"params": current_params, "batch_stats": batch_stats}, x,
                train=True, mutable=["batch_stats"],
            )
            policy_loss = -jnp.mean(jnp.sum(target_policy * jax.nn.log_softmax(logits), axis=1))
            value_loss = jnp.mean(jnp.square(value - target_value))
            return policy_loss + value_loss, (updated["batch_stats"], policy_loss, value_loss)
        (loss, (new_stats, ploss, vloss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        optimizer = optax.adamw(learning_rate, weight_decay=1e-4)
        changes, state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, changes), new_stats, state, loss, ploss, vloss

    configuration = {"base_checkpoint": str(source), "base_sha256": source_hash,
                     "updates": args.updates, "batch_size": args.batch_size, "seed": args.seed,
                     "learning_rates": [0.001, 0.0001], "optimizer": "AdamW",
                     "weight_decay": 0.0001, "optimizer_state": "fresh in both conditions",
                     "training_games": len(candidates), "training_position_labels": training_labels,
                     "data_cutoff_method": "files with mtime <= cycle50 checkpoint mtime; last 1000",
                     "evaluation_positions": len(labels), "evaluation_games_disjoint": True,
                     "learning_rate_schedule": "constant for this short run; original schedule first change is 50000 updates",
                     "output_dir": str(root)}
    (root / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    emit({"event": "prepared", **configuration})
    baseline = evaluate(base)
    emit({"event": "baseline", "evaluation": baseline})
    results = {}
    for name, learning_rate in (("A-lr-0.001", 0.001), ("B-lr-0.0001", 0.0001)):
        destination = root / name
        destination.mkdir()
        params, batch_stats = base["params"], base["batch_stats"]
        opt_state = optax.adamw(learning_rate, weight_decay=1e-4).init(params)
        replay = ReplayBuffer(sorted(data_dir.glob("*.npz")), 64, seed=args.seed)
        batch_hash = hashlib.sha256()
        history = [{"step": 0, "evaluation": baseline}]
        metrics = []
        for step in range(1, args.updates + 1):
            x, target_policy, target_value = replay.sample(args.batch_size)
            for batch in (x, target_policy, target_value):
                batch_hash.update(batch.tobytes())
            params, batch_stats, opt_state, loss, ploss, vloss = update(
                params, batch_stats, opt_state, jnp.asarray(x), jnp.asarray(target_policy),
                jnp.asarray(target_value), jnp.asarray(learning_rate, dtype=jnp.float32),
            )
            row = [float(loss), float(ploss), float(vloss)]
            assert np.isfinite(row).all(), f"Nonfinite training losses at {name}/{step}"
            metrics.append(row)
            if step % 16 == 0 or step == args.updates:
                emit({"event": "training", "condition": name, "step": step,
                      "losses_last16": np.mean(metrics[-16:], axis=0).tolist()})
            if step % 32 == 0 or step == args.updates:
                variables = {"params": params, "batch_stats": batch_stats}
                evaluation = evaluate(variables)
                history.append({"step": step, "evaluation": evaluation})
                save_variables(destination / f"step-{step:06d}.msgpack", variables)
                emit({"event": "evaluation", "condition": name, "step": step, "evaluation": evaluation})
        save_bytes_atomic(destination / "training_state.msgpack", serialization.to_bytes({
            "variables": {"params": params, "batch_stats": batch_stats},
            "opt_state": opt_state, "step": np.asarray(args.updates, dtype=np.int64),
        }))
        result = {"learning_rate": learning_rate, "batch_sha256": batch_hash.hexdigest(),
                  "training_losses": metrics, "history": history}
        (destination / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results[name] = result
    same_batches = len({result["batch_sha256"] for result in results.values()}) == 1
    assert same_batches, "Training batches differed between conditions"
    assert digest(source) == source_hash, "Source checkpoint was modified"
    summary = {"configuration": configuration, "identical_training_batches": same_batches,
               "base_checkpoint_unchanged": True, "baseline": baseline,
               "final": {name: result["history"][-1] for name, result in results.items()}}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit({"event": "completed", "output_dir": str(root), "identical_training_batches": same_batches})


if __name__ == "__main__":
    main()
