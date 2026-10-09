"""Small policy test for twofold/threefold repetition handling."""

import chess

from azchess.mcts import MCTS, Node


def policy_for(board: chess.Board) -> dict[chess.Move, float]:
    root = Node()
    root.children = {move: Node(1.0) for move in board.legal_moves}
    for child in root.children.values():
        child.visit_count = 10
    agent = object.__new__(MCTS)
    agent.search = lambda _board: root
    moves, probabilities, _ = agent.policy(board, temperature=1.0)
    return dict(zip(moves, probabilities))


twofold_board = chess.Board()
for uci in ("g1f3", "g8f6", "f3g1"):
    twofold_board.push_uci(uci)
twofold_policy = policy_for(twofold_board)
assert twofold_policy[chess.Move.from_uci("f6g8")] == 0.0

threefold_board = chess.Board()
for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
    threefold_board.push_uci(uci)
threefold_policy = policy_for(threefold_board)
assert threefold_policy[chess.Move.from_uci("f6g8")] == 0.0

reversal_board = chess.Board()
for uci in ("g1f3", "b8c6"):
    reversal_board.push_uci(uci)
reversal_policy = policy_for(reversal_board)
assert reversal_policy[chess.Move.from_uci("f3g1")] < reversal_policy[
    chess.Move.from_uci("a2a3")
]
print("REPETITION_POLICY_OK")
