from __future__ import annotations

import chess
import numpy as np

BOARD_PLANES = 20
ACTION_PLANES = 73
ACTION_SIZE = 64 * ACTION_PLANES

QUEEN_DIRECTIONS = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (-1, 1), (1, -1), (-1, -1),
)
KNIGHT_DIRECTIONS = (
    (1, 2), (2, 1), (-1, 2), (-2, 1),
    (1, -2), (2, -1), (-1, -2), (-2, -1),
)
UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def encode_board(board: chess.Board) -> np.ndarray:
    state = np.zeros((8, 8, BOARD_PLANES), dtype=np.float32)
    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + piece.piece_type - 1
        state[chess.square_rank(square), chess.square_file(square), plane] = 1.0
    state[:, :, 12] = float(board.turn == chess.WHITE)
    state[:, :, 13] = float(board.has_kingside_castling_rights(chess.WHITE))
    state[:, :, 14] = float(board.has_queenside_castling_rights(chess.WHITE))
    state[:, :, 15] = float(board.has_kingside_castling_rights(chess.BLACK))
    state[:, :, 16] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        state[chess.square_rank(board.ep_square), chess.square_file(board.ep_square), 17] = 1.0
    # 同じ駒配置でも反復回数によって意味が異なるため、盤面全体の履歴情報として渡す。
    state[:, :, 18] = float(board.is_repetition(2))
    state[:, :, 19] = float(board.is_repetition(3))
    return state


def move_to_action(board: chess.Board, move: chess.Move) -> int:
    from_file, from_rank = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    to_file, to_rank = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    df, dr = to_file - from_file, to_rank - from_rank

    if move.promotion in UNDERPROMOTIONS:
        forward = 1 if board.turn == chess.WHITE else -1
        if dr != forward or df not in (-1, 0, 1):
            raise ValueError(f"Invalid underpromotion: {move}")
        plane = 64 + (-1, 0, 1).index(df) * 3 + UNDERPROMOTIONS.index(move.promotion)
    elif (df, dr) in KNIGHT_DIRECTIONS:
        plane = 56 + KNIGHT_DIRECTIONS.index((df, dr))
    else:
        distance = max(abs(df), abs(dr))
        if distance < 1:
            raise ValueError(f"Invalid move: {move}")
        unit = (df // distance, dr // distance)
        if unit not in QUEEN_DIRECTIONS or distance > 7:
            raise ValueError(f"Unencodable move: {move}")
        plane = QUEEN_DIRECTIONS.index(unit) * 7 + distance - 1
    return move.from_square * ACTION_PLANES + plane


def legal_action_map(board: chess.Board) -> dict[int, chess.Move]:
    return {move_to_action(board, move): move for move in board.legal_moves}


def terminal_value(board: chess.Board) -> float | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0
