from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from azchess.checkpoint import load_variables, save_bytes_atomic, save_variables
from azchess.game import ACTION_SIZE, BOARD_PLANES
from azchess.data_cache import DataCache


def create_optimizer(learning_rate: float = 1e-3, constant: bool = False):
    if constant:
        return optax.adamw(learning_rate, weight_decay=1e-4)
    schedule = optax.piecewise_constant_schedule(
        init_value=learning_rate,
        boundaries_and_scales={50_000: 0.3, 150_000: 0.3, 300_000: 0.3},
    )
    return optax.adamw(schedule, weight_decay=1e-4)


class ReplayBuffer:
    def __init__(self, files: list[Path], cache_games: int, seed: int | None = None, cache_mb: int = 512) -> None:
        self.files = files
        self.cache_games = cache_games
        self.rng = np.random.default_rng(seed)
        self.cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.cache_bytes=0
        self.max_cache_bytes=max(1,cache_mb)*1024*1024
        lengths = []
        metadata=DataCache(files[0].parent)
        for path in files:
            lengths.append(metadata.get(path)['positions'])
        metadata.flush()
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.probabilities = self.lengths / self.lengths.sum()

    def load_game(self, index: int) -> dict[str, np.ndarray]:
        cached = self.cache.pop(index, None)
        if cached is not None:
            self.cache[index] = cached
            return cached
        with np.load(self.files[index]) as item:
            game = {name: item[name] for name in (
                "states", "policy_offsets", "policy_indices", "policy_values", "values"
            )}
        self.cache[index] = game
        self.cache_bytes+=sum(array.nbytes for array in game.values())
        while len(self.cache) > self.cache_games or self.cache_bytes > self.max_cache_bytes:
            _,discarded=self.cache.popitem(last=False)
            self.cache_bytes-=sum(array.nbytes for array in discarded.values())
        return game

    def sample(self, batch_size: int):
        states = np.empty((batch_size, 8, 8, BOARD_PLANES), dtype=np.float32)
        policies = np.zeros((batch_size, ACTION_SIZE), dtype=np.float32)
        values = np.empty(batch_size, dtype=np.float32)
        game_indices = self.rng.choice(
            len(self.files), size=batch_size, replace=True, p=self.probabilities
        )
        for batch_index, game_index_raw in enumerate(game_indices):
            game_index = int(game_index_raw)
            game = self.load_game(game_index)
            position = int(self.rng.integers(0, len(game["values"])))
            states[batch_index] = game["states"][position]
            start = int(game["policy_offsets"][position])
            end = int(game["policy_offsets"][position + 1])
            indices = game["policy_indices"][start:end].astype(np.int32)
            policies[batch_index, indices] = game["policy_values"][start:end]
            values[batch_index] = game["values"][position]
        return states, policies, values


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scaled AlphaZero v2")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-games", type=int, default=1000)
    parser.add_argument("--cache-games", type=int, default=256)
    parser.add_argument("--cache-mb", type=int, default=512)
    parser.add_argument("--data-dir", type=Path, default=Path("selfplay_data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.msgpack"))
    parser.add_argument(
        "--training-state", type=Path, default=Path("checkpoints/training_state.msgpack")
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--constant-learning-rate", action="store_true", help="Use for optimizer states from constant-rate experiments")
    parser.add_argument("--no-prefetch", action="store_true", help="Disable overlapping replay loading with training")
    args = parser.parse_args()
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    files = sorted(args.data_dir.glob("*.npz"))[-args.max_games :]
    if not files:
        raise SystemExit("No v2 self-play data. Run selfplay.py first.")

    model, latest_variables = load_variables(args.checkpoint)
    optimizer = create_optimizer(args.learning_rate, constant=args.constant_learning_rate)
    template = {
        "variables": latest_variables,
        "opt_state": optimizer.init(latest_variables["params"]),
        "step": np.asarray(0, dtype=np.int64),
    }
    if args.training_state.exists():
        bundle = serialization.from_bytes(template, args.training_state.read_bytes())
    else:
        bundle = template
    variables = bundle["variables"]
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    opt_state = bundle["opt_state"]
    start_step = int(bundle["step"])
    replay = ReplayBuffer(files, max(1, args.cache_games), seed=args.seed, cache_mb=args.cache_mb)
    print(
        f"Replay: {len(files)} games, {int(replay.lengths.sum())} positions; "
        f"updates={args.steps}, starting_step={start_step}",
        flush=True,
    )

    @jax.jit
    def train_step(params, batch_stats, opt_state, x, target_policy, target_value):
        def loss_fn(current_params):
            (logits, predicted_value), updates = model.apply(
                {"params": current_params, "batch_stats": batch_stats},
                x,
                train=True,
                mutable=["batch_stats"],
            )
            policy_loss = -jnp.mean(
                jnp.sum(target_policy * jax.nn.log_softmax(logits), axis=1)
            )
            value_loss = jnp.mean(jnp.square(predicted_value - target_value))
            return policy_loss + value_loss, (
                updates["batch_stats"],
                policy_loss,
                value_loss,
            )

        (loss, (new_stats, policy_loss, value_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_stats, new_opt_state, loss, policy_loss, value_loss

    recent_metrics = []
    # A single producer owns ReplayBuffer/RNG. Samples stay in the original
    # order; only loading the next batch overlaps the current device update.
    with ThreadPoolExecutor(max_workers=1) as loader:
        pending = None if args.no_prefetch or args.steps <= 0 else loader.submit(replay.sample, args.batch_size)
        for local_step in range(args.steps):
            states, policies, values = replay.sample(args.batch_size) if args.no_prefetch else pending.result()
            if not args.no_prefetch and local_step + 1 < args.steps:
                pending = loader.submit(replay.sample, args.batch_size)
            params, batch_stats, opt_state, loss, policy_loss, value_loss = train_step(
                params,
                batch_stats,
                opt_state,
                jnp.asarray(states),
                jnp.asarray(policies),
                jnp.asarray(values),
            )
            recent_metrics.append((loss, policy_loss, value_loss))
            if (local_step + 1) % 16 == 0 or local_step + 1 == args.steps:
                mean = np.mean(np.asarray(jax.device_get(recent_metrics[-16:]),dtype=np.float64), axis=0)
                recent_metrics.clear()
                print(
                    f"step={start_step + local_step + 1} loss={mean[0]:.4f} "
                    f"policy={mean[1]:.4f} value={mean[2]:.4f}",
                    flush=True,
                )

    variables = {"params": params, "batch_stats": batch_stats}
    bundle = {
        "variables": variables,
        "opt_state": opt_state,
        "step": np.asarray(start_step + args.steps, dtype=np.int64),
    }
    save_bytes_atomic(args.training_state, serialization.to_bytes(bundle))
    save_variables(args.checkpoint, variables)
    print(f"Saved: {args.checkpoint}; training_step={start_step + args.steps}", flush=True)


if __name__ == "__main__":
    main()
