from __future__ import annotations

import chess
import numpy as np


HISTORY_LENGTH = 8
PLANES_PER_HISTORY = 14
BOARD_PLANES = HISTORY_LENGTH * PLANES_PER_HISTORY + 7
ACTION_PLANES = 73
ACTION_SIZE = 64 * ACTION_PLANES

QUEEN_DIRECTIONS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
)
KNIGHT_DIRECTIONS = (
    (1, 2),
    (2, 1),
    (-1, 2),
    (-2, 1),
    (1, -2),
    (2, -1),
    (-1, -2),
    (-2, -1),
)
UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def canonical_square(square: chess.Square, player: chess.Color) -> chess.Square:
    if player == chess.WHITE:
        return square
    return chess.square(chess.square_file(square), 7 - chess.square_rank(square))


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode the current position and seven predecessors from side-to-move view."""
    state = np.zeros((8, 8, BOARD_PLANES), dtype=np.float32)
    player = board.turn
    historical = board.copy(stack=True)
    for history_index in range(HISTORY_LENGTH):
        base = history_index * PLANES_PER_HISTORY
        for square, piece in historical.piece_map().items():
            oriented = canonical_square(square, player)
            owner_offset = 0 if piece.color == player else 6
            plane = base + owner_offset + piece.piece_type - 1
            state[chess.square_rank(oriented), chess.square_file(oriented), plane] = 1.0
        state[:, :, base + 12] = float(historical.is_repetition(2))
        state[:, :, base + 13] = float(historical.is_repetition(3))
        if not historical.move_stack:
            break
        historical.pop()

    state[:, :, 112] = float(player == chess.WHITE)
    state[:, :, 113] = min(len(board.move_stack), 512) / 512.0
    state[:, :, 114] = float(board.has_kingside_castling_rights(player))
    state[:, :, 115] = float(board.has_queenside_castling_rights(player))
    state[:, :, 116] = float(board.has_kingside_castling_rights(not player))
    state[:, :, 117] = float(board.has_queenside_castling_rights(not player))
    state[:, :, 118] = min(board.halfmove_clock, 100) / 100.0
    return state


def move_to_action(board: chess.Board, move: chess.Move) -> int:
    source = canonical_square(move.from_square, board.turn)
    target = canonical_square(move.to_square, board.turn)
    from_file, from_rank = chess.square_file(source), chess.square_rank(source)
    to_file, to_rank = chess.square_file(target), chess.square_rank(target)
    df, dr = to_file - from_file, to_rank - from_rank

    if move.promotion in UNDERPROMOTIONS:
        if dr != 1 or df not in (-1, 0, 1):
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
    return source * ACTION_PLANES + plane


def legal_action_map(board: chess.Board) -> dict[int, chess.Move]:
    moves = list(board.legal_moves)
    mapping = {move_to_action(board, move): move for move in moves}
    if len(mapping) != len(moves):
        raise RuntimeError("Two legal moves mapped to the same policy action")
    return mapping


DRAW_RULE = "az_actual_threefold_50move_v1"


def game_outcome(board: chess.Board) -> chess.Outcome | None:
    """AlphaZero rules: actual three repetitions/100 quiet plies are automatic.

    Normal decisive endings take precedence, including mate on the 100th ply.
    can_claim_* is deliberately avoided: it includes prospective next moves.
    """
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        return outcome
    if board.is_repetition(3):
        return chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    if board.is_fifty_moves():
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    return None


def terminal_value(board: chess.Board) -> float | None:
    outcome = game_outcome(board)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0
