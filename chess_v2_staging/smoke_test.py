"""Fast structural checks that do not create training data."""

import random
from pathlib import Path

import chess

from azchess.game import ACTION_SIZE, BOARD_PLANES, encode_board, legal_action_map


def main() -> None:
    rng = random.Random(1234)
    checked = 0
    for _ in range(20):
        board = chess.Board()
        for _ in range(80):
            if board.is_game_over(claim_draw=True):
                break
            encoded = encode_board(board)
            assert encoded.shape == (8, 8, BOARD_PLANES)
            mapping = legal_action_map(board)
            assert len(mapping) == board.legal_moves.count()
            assert all(0 <= action < ACTION_SIZE for action in mapping)
            board.push(rng.choice(list(board.legal_moves)))
            checked += 1
    print(f"smoke_test ok: {checked} positions")
    checkpoint = Path("checkpoints/latest.msgpack")
    if checkpoint.exists():
        from azchess.checkpoint import load_variables
        from azchess.mcts import MCTS

        model, variables = load_variables(checkpoint)
        move = MCTS(model, variables, simulations=1).choose_move(chess.Board())
        assert move in chess.Board().legal_moves
        print(f"model_inference_ok: {move.uci()}")


if __name__ == "__main__":
    main()
