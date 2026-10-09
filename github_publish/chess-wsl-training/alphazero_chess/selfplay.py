from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import chess
import numpy as np

from azchess.checkpoint import load_variables
from azchess.game import encode_board
from azchess.mcts import MCTS


MATERIAL_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def material_value_for_white(board: chess.Board) -> float:
    difference = 0.0
    for piece_type, value in MATERIAL_VALUES.items():
        difference += value * len(board.pieces(piece_type, chess.WHITE))
        difference -= value * len(board.pieces(piece_type, chess.BLACK))
    # 打ち切りは勝敗より十分小さく評価しつつ、駒価値の差は残す。
    # 例: ルーク1個得=0.10、クイーン1個得=0.18、10点以上=上限0.20。
    return float(np.clip(difference / 50.0, -0.2, 0.2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AlphaZero self-play data")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.msgpack"))
    args = parser.parse_args()
    model, variables = load_variables(args.checkpoint)
    agent = MCTS(model, variables, simulations=args.simulations, root_noise=True)
    output_dir = Path("selfplay_data")
    output_dir.mkdir(exist_ok=True)

    for game_index in range(args.games):
        board = chess.Board()
        samples = []
        played_moves: list[str] = []
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < args.max_plies:
            if len(board.move_stack) < 30:
                temperature = 1.0
            elif board.halfmove_clock >= 30:
                temperature = 0.8
            else:
                temperature = 0.5
            moves, probabilities, dense_policy = agent.policy(board, temperature)
            chosen = int(np.random.choice(len(moves), p=probabilities))
            samples.append((encode_board(board), dense_policy, board.turn))
            selected_move = moves[chosen]
            played_moves.append(selected_move.uci())
            board.push(selected_move)
        outcome = board.outcome(claim_draw=True)
        winner = outcome.winner if outcome else None
        states = np.stack([sample[0] for sample in samples])
        policies = np.stack([sample[1] for sample in samples])
        repetition_draw = outcome is not None and outcome.termination in (
            chess.Termination.THREEFOLD_REPETITION,
            chess.Termination.FIVEFOLD_REPETITION,
        )
        adjudication = ""
        if outcome is None:
            white_value = material_value_for_white(board)
            target_values = [
                white_value if player == chess.WHITE else -white_value
                for _, _, player in samples
            ]
            adjudication = f" material_white={white_value:+.2f}"
        elif repetition_draw:
            # 反復を成立させた最後の手の側へ小さな負値を与え、無意味な往復を抑える。
            repeating_player = not board.turn
            target_values = [
                -0.3 if player == repeating_player else 0.3
                for _, _, player in samples
            ]
            adjudication = " repetition_penalty=0.30"
        else:
            target_values = [
                0.0 if winner is None else (1.0 if winner == player else -1.0)
                for _, _, player in samples
            ]
        values = np.asarray(target_values, dtype=np.float32)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = output_dir / f"game-{stamp}.npz"
        termination = outcome.termination.name if outcome else "MAX_PLIES"
        np.savez_compressed(
            output,
            states=states,
            policies=policies,
            values=values,
            moves=np.asarray(played_moves, dtype="<U5"),
            final_fen=np.asarray(board.fen()),
            termination=np.asarray(termination),
        )
        print(
            f"Game {game_index + 1}: {board.result(claim_draw=True)} "
            f"termination={termination}{adjudication} plies={len(samples)} -> {output}"
        )


if __name__ == "__main__":
    main()
