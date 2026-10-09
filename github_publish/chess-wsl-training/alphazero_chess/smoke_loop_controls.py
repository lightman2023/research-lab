"""Smoke checks for cutoff scoring and diagnostic self-play metadata."""

from pathlib import Path

import chess
import numpy as np

from selfplay import material_value_for_white


board = chess.Board()
assert material_value_for_white(board) == 0.0
board.remove_piece_at(chess.D8)
assert np.isclose(material_value_for_white(board), 0.18)
board = chess.Board()
board.remove_piece_at(chess.D1)
assert np.isclose(material_value_for_white(board), -0.18)

files = sorted(Path("selfplay_data").glob("*.npz"))
if files:
    with np.load(files[-1]) as data:
        assert {"states", "policies", "values", "moves", "final_fen", "termination"} <= set(data.files)
        assert len(data["moves"]) == len(data["values"])

print("LOOP_CONTROLS_OK")
