"""Inspect the value predictor and MCTS without playing new games."""

import json
import sys
from collections import Counter
from pathlib import Path

import chess.pgn
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/home/user/chess/alphazero_chess_v2")
from azchess.checkpoint import load_variables
from azchess.mcts import MCTS


def main():
    project = Path("/home/user/chess/alphazero_chess_v2")
    files = sorted((project / "experiments/raw-visits-v1/selfplay_data").glob("*.npz"))
    outcomes = Counter()
    recent_outcomes = Counter()
    label_counts = Counter()
    pools = {0: [], 1: [], -1: []}
    rng = np.random.default_rng(20260928)
    for file_index, path in enumerate(files):
        with np.load(path) as data:
            termination = str(data["termination"])
            outcomes[termination] += 1
            if file_index < len(files) - 1000:
                continue
            recent_outcomes[termination] += 1
            values = data["values"]
            states = data["states"]
            for label in pools:
                positions = np.flatnonzero(values == label)
                label_counts[label] += len(positions)
                if len(positions):
                    chosen = rng.choice(positions, size=min(4, len(positions)), replace=False)
                    pools[label].extend(states[chosen].astype(np.float32))

    model, variables = load_variables(project / "experiments/raw-visits-v1/checkpoints/saved/cycle-000400.msgpack")
    predict = jax.jit(lambda x: model.apply(variables, x, train=False)[1])
    value_stats = {}
    for label, states in pools.items():
        selected = rng.choice(len(states), size=min(256, len(states)), replace=False)
        sample = np.stack([states[index] for index in selected])
        predictions = np.concatenate([
            np.asarray(predict(jnp.asarray(sample[start:start + 64]))).reshape(-1)
            for start in range(0, len(sample), 64)
        ])
        value_stats[label] = {
            "sample_positions": len(predictions),
            "prediction_mean": float(predictions.mean()),
            "prediction_abs_mean": float(np.abs(predictions).mean()),
            "prediction_min": float(predictions.min()),
            "prediction_max": float(predictions.max()),
            "mse": float(np.mean((predictions - label) ** 2)),
            "zero_prediction_mse": float(label ** 2),
        }
    print("VALUE_STATS", json.dumps(value_stats), flush=True)

    with (project / "arena_results/match-20260927-214420-d7f1a96d.pgn").open() as stream:
        for _ in range(11):
            game = chess.pgn.read_game(stream)
    moves = list(game.mainline_moves())
    search_stats = []
    for ply in (13, 19):
        board = game.board()
        for move in moves[:ply]:
            board.push(move)
        previous_own_move = board.move_stack[-2]
        for simulations in (50, 400):
            agent = MCTS(model, variables, simulations=simulations)
            _, network_value = agent.evaluate(board)
            root = agent.search(board)
            rows = []
            for move, child in sorted(root.children.items(), key=lambda item: item[1].visit_count, reverse=True):
                if move is None:
                    rows.append({"move": "claim_draw", "visits": child.visit_count, "q": 0.0})
                    continue
                after = board.copy(stack=True)
                after.push(move)
                rows.append({
                    "move": move.uci(), "san": board.san(move),
                    "prior": child.prior, "visits": child.visit_count,
                    "q": -child.value,
                    "reversal": move.from_square == previous_own_move.to_square and move.to_square == previous_own_move.from_square,
                    "twofold_after": after.is_repetition(2),
                    "claimable_after": after.can_claim_draw(),
                })
            item = {"game": 11, "ply": ply, "simulations": simulations,
                    "network_value": network_value, "root_value": root.value, "moves": rows}
            search_stats.append(item)
            print("SEARCH", json.dumps({**item, "moves": rows[:8]}), flush=True)

    result = {"games": len(files), "outcomes": dict(outcomes),
              "recent_games": min(1000, len(files)), "recent_outcomes": dict(recent_outcomes),
              "recent_position_labels": dict(label_counts), "value_samples": value_stats,
              "searches": search_stats}
    output = Path(__file__).with_name("draw_cycle_diagnosis.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
