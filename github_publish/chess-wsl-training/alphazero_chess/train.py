from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from azchess.augmentation import mirror_policies, mirror_states
from azchess.checkpoint import load_variables, save_variables


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on AlphaZero self-play data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--no-color-augmentation", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.msgpack"))
    args = parser.parse_args()

    files = sorted(Path("selfplay_data").glob("*.npz"))
    if args.max_files > 0:
        files = files[-args.max_files :]
    if not files:
        raise SystemExit("No self-play data. Run selfplay.py first.")
    arrays = [np.load(path) for path in files]
    states = np.concatenate([item["states"] for item in arrays])
    policies = np.concatenate([item["policies"] for item in arrays])
    values = np.concatenate([item["values"] for item in arrays])
    use_augmentation = not args.no_color_augmentation
    effective_positions = len(states) * (2 if use_augmentation else 1)
    print(
        f"Training data: {len(files)} games, {len(states)} positions, "
        f"color_augmented={use_augmentation}, effective={effective_positions}",
        flush=True,
    )
    model, variables = load_variables(args.checkpoint)
    params, batch_stats = variables["params"], variables["batch_stats"]
    optimizer = optax.adamw(args.learning_rate, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, batch_stats, opt_state, x, target_policy, target_value):
        def loss_fn(current_params):
            (logits, predicted_value), updates = model.apply(
                {"params": current_params, "batch_stats": batch_stats},
                x,
                train=True,
                mutable=["batch_stats"],
            )
            policy_loss = -jnp.mean(jnp.sum(target_policy * jax.nn.log_softmax(logits), axis=1))
            value_loss = jnp.mean(jnp.square(predicted_value - target_value))
            return policy_loss + value_loss, (updates["batch_stats"], policy_loss, value_loss)

        (loss, (new_stats, policy_loss, value_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_stats, new_opt_state, loss, policy_loss, value_loss

    rng = np.random.default_rng()
    for epoch in range(args.epochs):
        order = rng.permutation(effective_positions)
        metrics = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            source_indices = indices % len(states)
            batch_states = states[source_indices].copy()
            batch_policies = policies[source_indices].copy()
            if use_augmentation:
                mirrored = indices >= len(states)
                if mirrored.any():
                    batch_states[mirrored] = mirror_states(batch_states[mirrored])
                    batch_policies[mirrored] = mirror_policies(batch_policies[mirrored])
            params, batch_stats, opt_state, *losses = train_step(
                params,
                batch_stats,
                opt_state,
                jnp.asarray(batch_states),
                jnp.asarray(batch_policies),
                jnp.asarray(values[source_indices]),
            )
            metrics.append([float(value) for value in losses])
        mean = np.mean(metrics, axis=0)
        print(f"epoch={epoch + 1} loss={mean[0]:.4f} policy={mean[1]:.4f} value={mean[2]:.4f}")
    save_variables(args.checkpoint, {"params": params, "batch_stats": batch_stats})
    print(f"Saved: {args.checkpoint}")


if __name__ == "__main__":
    main()
