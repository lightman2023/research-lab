"""Validate color-swapped state and policy augmentation."""

import chess
import numpy as np

from azchess.augmentation import (
    MIRRORED_ACTIONS,
    mirror_action_index,
    mirror_policies,
    mirror_square,
    mirror_states,
)
from azchess.game import ACTION_SIZE, encode_board, move_to_action


board = chess.Board()
for uci in ("e2e4", "c7c5", "g1f3", "d7d6"):
    board.push_uci(uci)
mirrored_board = board.mirror()
np.testing.assert_array_equal(
    mirror_states(encode_board(board)[None, ...])[0], encode_board(mirrored_board)
)

for move in board.legal_moves:
    mirrored_move = chess.Move(
        mirror_square(move.from_square),
        mirror_square(move.to_square),
        promotion=move.promotion,
    )
    assert mirror_action_index(move_to_action(board, move)) == move_to_action(
        mirrored_board, mirrored_move
    )

assert len(np.unique(MIRRORED_ACTIONS)) == ACTION_SIZE
policy = np.arange(ACTION_SIZE, dtype=np.float32)[None, :]
np.testing.assert_array_equal(mirror_policies(mirror_policies(policy)), policy)
print("COLOR_AUGMENTATION_OK")
