from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import chess
import numpy as np

from azchess.checkpoint import load_variables
from azchess.game import encode_board, move_to_action
from azchess.mcts import MCTS


REPETITION_WEIGHT = 0.0
REVERSAL_WEIGHT = 0.01


def training_policy(agent: MCTS, board: chess.Board, temperature: float):
    """Build a policy target and a separate move-sampling distribution."""
    root = agent.search(board)
    moves = list(root.children)
    if not moves:
        raise ValueError("Cannot choose a move from a terminal position")
    visits = np.asarray(
        [root.children[move].visit_count for move in moves], dtype=np.float64
    )
    weights = np.ones_like(visits)
    previous_own_move = board.move_stack[-2] if len(board.move_stack) >= 2 else None
    for index, move in enumerate(moves):
        candidate = board.copy(stack=True)
        candidate.push(move)
        if candidate.is_repetition(2):
            weights[index] = REPETITION_WEIGHT
        elif (
            previous_own_move is not None
            and move.from_square == previous_own_move.to_square
            and move.to_square == previous_own_move.from_square
        ):
            weights[index] = REVERSAL_WEIGHT

    adjusted_visits = visits * weights
    if adjusted_visits.sum() <= 0:
        adjusted_visits = np.ones_like(visits)
    target = adjusted_visits / adjusted_visits.sum()
    sampling_weights = np.power(adjusted_visits, 1.0 / temperature)
    sampling = sampling_weights / sampling_weights.sum()
    indices = np.asarray(
        [move_to_action(board, move) for move in moves], dtype=np.uint16
    )
    return moves, sampling, indices, target


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v2 AlphaZero self-play data")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.msgpack"))
    parser.add_argument("--output-dir", type=Path, default=Path("selfplay_data_loopfix"))
    args = parser.parse_args()
    model, variables = load_variables(args.checkpoint)
    agent = MCTS(model, variables, simulations=args.simulations, root_noise=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for game_index in range(args.games):
        board = chess.Board()
        states: list[np.ndarray] = []
        policy_indices: list[np.ndarray] = []
        policy_values: list[np.ndarray] = []
        policy_offsets = [0]
        players: list[chess.Color] = []
        played_moves: list[str] = []
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < args.max_plies:
            temperature = 1.0 if len(board.move_stack) < 30 else 0.8
            moves, sampling, indices, target = training_policy(
                agent, board, temperature
            )
            selected = int(np.random.choice(len(moves), p=sampling))
            states.append(encode_board(board).astype(np.float16))
            policy_indices.append(indices)
            policy_values.append(target.astype(np.float16))
            policy_offsets.append(policy_offsets[-1] + len(indices))
            players.append(board.turn)
            move = moves[selected]
            played_moves.append(move.uci())
            board.push(move)
            if len(board.move_stack) % 10 == 0:
                print(
                    f"Game {game_index + 1} progress: "
                    f"plies={len(board.move_stack)} last={move.uci()}",
                    flush=True,
                )

        outcome = board.outcome(claim_draw=True)
        winner = outcome.winner if outcome is not None else None
        if outcome is None:
            piece_values = {
                chess.PAWN: 1.0,
                chess.KNIGHT: 3.0,
                chess.BISHOP: 3.0,
                chess.ROOK: 5.0,
                chess.QUEEN: 9.0,
            }
            material_white = sum(
                value
                * (
                    len(board.pieces(piece_type, chess.WHITE))
                    - len(board.pieces(piece_type, chess.BLACK))
                )
                for piece_type, value in piece_values.items()
            )
            white_value = float(np.clip(material_white / 50.0, -0.2, 0.2))
            values = np.asarray(
                [white_value if player == chess.WHITE else -white_value for player in players],
                dtype=np.float32,
            )
        else:
            values = np.asarray(
                [
                    0.0 if winner is None else (1.0 if winner == player else -1.0)
                    for player in players
                ],
                dtype=np.float32,
            )
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = args.output_dir / f"game-{stamp}.npz"
        termination = outcome.termination.name if outcome is not None else "MAX_PLIES_DRAW"
        np.savez_compressed(
            output,
            states=np.stack(states),
            policy_offsets=np.asarray(policy_offsets, dtype=np.int32),
            policy_indices=np.concatenate(policy_indices).astype(np.uint16),
            policy_values=np.concatenate(policy_values).astype(np.float16),
            values=values,
            moves=np.asarray(played_moves, dtype="<U5"),
            final_fen=np.asarray(board.fen()),
            termination=np.asarray(termination),
            policy_target_kind=np.asarray("repetition_adjusted_visits_v2"),
        )
        result = board.result(claim_draw=True) if outcome is not None else "1/2-1/2"
        print(
            f"Game {game_index + 1}: {result} termination={termination} "
            f"plies={len(states)} -> {output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
