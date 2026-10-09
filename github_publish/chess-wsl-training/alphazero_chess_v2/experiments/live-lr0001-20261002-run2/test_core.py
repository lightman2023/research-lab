import json
import random
from pathlib import Path
import sys
sys.path.insert(0, '/home/user/chess/alphazero_chess_v2')
import chess
import numpy as np
from azchess.game import game_outcome, terminal_value, encode_board, legal_action_map, move_to_action
from azchess.mcts import MCTS

checks = []
def check(name, condition):
    assert condition, name
    checks.append(name)

board = chess.Board()
for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1']:
    board.push_uci(uci)
check('prospective_third_not_terminal', board.can_claim_threefold_repetition() and game_outcome(board) is None)
before = board.copy(stack=True)
board.push_uci('f6g8')
check('actual_third_automatic', game_outcome(board).termination == chess.Termination.THREEFOLD_REPETITION)
check('history_preserved_for_search', terminal_value(board.copy(stack=True)) == 0 and terminal_value(chess.Board(board.fen())) is None)
check('history_repetition_input', encode_board(board)[0, 0, 13] == 1 and encode_board(chess.Board(board.fen()))[0, 0, 13] == 0)
quiet = chess.Board('7k/8/8/8/8/8/8/R5K1 w - - 99 51')
check('prospective_50_not_terminal', quiet.can_claim_fifty_moves() and game_outcome(quiet) is None)
quiet.push_uci('a1a2')
check('actual_50_automatic', game_outcome(quiet).termination == chess.Termination.FIFTY_MOVES)
mate = chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 99 51')
mate.push_uci('g6g7')
check('mate_precedes_50_and_loser_sign', game_outcome(mate).winner == chess.WHITE and terminal_value(mate) == -1)
black = chess.Board()
for uci in ['f2f3', 'e7e5', 'g2g4', 'd8h4']:
    black.push_uci(uci)
check('black_mate_loser_sign', terminal_value(black) == -1 and game_outcome(black).winner == chess.BLACK)
check('stalemate', terminal_value(chess.Board('7k/5K2/6Q1/8/8/8/8/8 b - - 0 1')) == 0)
check('insufficient_material', terminal_value(chess.Board('7k/8/8/8/8/8/8/6K1 w - - 0 1')) == 0)

class RulesOnly(MCTS):
    def __init__(self, sims=100):
        self.simulations, self.root_noise = sims, False
        self.c_init, self.c_base = 1.25, 19652.
    def evaluate(self, board):
        moves = list(board.legal_moves)
        return {m: 1 / len(moves) for m in moves}, 0.

agent = RulesOnly(200)
position = chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
root = agent.search(position)
child = root.children[chess.Move.from_uci('g6g7')]
check('mate_child_negative_root_positive', child.value == -1 and root.value > 0)
check('mcts_selects_mate', agent.choose_action(position).uci() == 'g6g7')
root = agent.search(before)
check('mcts_sees_repetition_child_as_draw', root.children[chess.Move.from_uci('f6g8')].value == 0)
check('terminal_root_no_expansion', not agent.search(board).children)

class KnownVisits(RulesOnly):
    def search(self, board):
        from azchess.mcts import Node
        root = Node()
        for move, visits in zip(list(board.legal_moves)[:2], [20, 10]):
            root.children[move] = Node()
            root.children[move].visit_count = visits
        return root
_, probabilities, _, target = KnownVisits().policy_and_target(chess.Board(), 0.25)
check('raw_visits_target_not_sharpened', np.allclose(target, [2/3, 1/3]) and np.allclose(probabilities, [16/17, 1/17]))

rng = random.Random(20261002)
count = 0
special = [chess.Board('7k/P7/8/8/8/8/6Kp/8 w - - 0 1'), chess.Board('6k1/8/8/8/8/8/p6P/7K b - - 0 1'), chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'), chess.Board('7k/8/8/3pP3/8/8/8/6K1 w - d6 0 1')]
for color,position in zip(['white','black'],special[:2]):
    check('all_underpromotions_'+color, {m.promotion for m in position.legal_moves if m.promotion} == {chess.QUEEN,chess.ROOK,chess.BISHOP,chess.KNIGHT})
for position in special:
    mapping = legal_action_map(position)
    check('special_action_map_' + str(count), len(mapping) == position.legal_moves.count() and all(0 <= a < 4672 for a in mapping))
    count += len(mapping)
for _ in range(10):
    position = chess.Board()
    for _ in range(80):
        mapping = legal_action_map(position)
        assert all(move_to_action(position, move) == action and 0 <= action < 4672 for action, move in mapping.items())
        count += len(mapping)
        if game_outcome(position) is not None:
            break
        position.push(rng.choice(list(position.legal_moves)))
checks.append(f'legal_action_maps_random_games_{count}_moves')
Path(__file__).with_name('core_test_results.json').write_text(json.dumps({'passed': len(checks), 'checks': checks}, indent=2))
print(json.dumps({'passed': len(checks), 'checks': checks}))
