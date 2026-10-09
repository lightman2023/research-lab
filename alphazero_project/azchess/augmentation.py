"""Color-swap augmentation for encoded chess states and AlphaZero actions."""

from __future__ import annotations

import chess
import numpy as np

from .game import ACTION_PLANES, ACTION_SIZE, KNIGHT_DIRECTIONS, QUEEN_DIRECTIONS


def mirror_square(square: chess.Square) -> chess.Square:
    return chess.square(chess.square_file(square), 7 - chess.square_rank(square))


def mirror_action_index(action: int) -> int:
    square, plane = divmod(action, ACTION_PLANES)
    mirrored_square = mirror_square(square)
    if plane < 56:
        direction_index, distance_index = divmod(plane, 7)
        df, dr = QUEEN_DIRECTIONS[direction_index]
        mirrored_direction = (df, -dr)
        mirrored_plane = QUEEN_DIRECTIONS.index(mirrored_direction) * 7 + distance_index
    elif plane < 64:
        df, dr = KNIGHT_DIRECTIONS[plane - 56]
        mirrored_plane = 56 + KNIGHT_DIRECTIONS.index((df, -dr))
    else:
        # 色と進行方向を同時に反転するため、昇格の左右方向と駒種は同じ。
        mirrored_plane = plane
    return mirrored_square * ACTION_PLANES + mirrored_plane


MIRRORED_ACTIONS = np.asarray(
    [mirror_action_index(action) for action in range(ACTION_SIZE)], dtype=np.int32
)


def mirror_states(states: np.ndarray) -> np.ndarray:
    flipped = states[:, ::-1, :, :]
    mirrored = flipped.copy()
    mirrored[..., 0:6] = flipped[..., 6:12]
    mirrored[..., 6:12] = flipped[..., 0:6]
    mirrored[..., 12] = 1.0 - flipped[..., 12]
    mirrored[..., 13] = flipped[..., 15]
    mirrored[..., 14] = flipped[..., 16]
    mirrored[..., 15] = flipped[..., 13]
    mirrored[..., 16] = flipped[..., 14]
    return mirrored


def mirror_policies(policies: np.ndarray) -> np.ndarray:
    mirrored = np.zeros_like(policies)
    mirrored[:, MIRRORED_ACTIONS] = policies
    return mirrored
