"""Focused checks for optional threefold claims and policy targets."""

import chess
import numpy as np

from azchess.game import terminal_value
from azchess.mcts import MCTS, Node


def repeated_knights(cycles: int, extra_moves: tuple[str, ...] = ()) -> chess.Board:
    board = chess.Board()
    for _ in range(cycles):
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(uci)
    for uci in extra_moves:
        board.push_uci(uci)
    return board


def main() -> None:
    claimable = repeated_knights(1, ("g1f3", "g8f6", "f3g1"))
    assert claimable.can_claim_threefold_repetition()
    assert not claimable.is_repetition(3)
    assert terminal_value(claimable) is None

    agent = MCTS(None, None, simulations=50)

    def uniform_evaluate(board):
        moves = list(board.legal_moves)
        return {move: 1.0 / len(moves) for move in moves}, 0.0

    agent.evaluate = uniform_evaluate
    root = agent.search(claimable)
    assert None in root.children
    assert len(root.children) == claimable.legal_moves.count() + 1
    assert sum(child.visit_count for child in root.children.values()) == 50

    moves = list(claimable.legal_moves)[:2]
    root = Node()
    root.children = {moves[0]: Node(), moves[1]: Node(), None: Node()}
    root.children[moves[0]].visit_count = 10
    root.children[moves[1]].visit_count = 5
    root.children[None].visit_count = 20
    agent.search = lambda board: root
    choice, indices, target = agent.decision_and_target(claimable, 0.0)
    assert choice is None
    assert len(indices) == 2
    np.testing.assert_allclose(target, [2 / 3, 1 / 3])

    forced = repeated_knights(4)
    assert forced.is_fivefold_repetition()
    assert terminal_value(forced) == 0.0
    print("optional draw claim checks passed")


if __name__ == "__main__":
    main()
