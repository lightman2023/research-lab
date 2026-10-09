from pathlib import Path
import shutil

root = Path('/home/user/chess/alphazero_chess_v2')
def edit(name, transform):
    path = root / name
    backup = path.with_name(path.name + '.pre-azrules-20261002')
    if backup.exists():
        raise RuntimeError(f'Already prepared: {backup}')
    shutil.copy2(path, backup)
    path.write_text(transform(path.read_text()), encoding='utf-8')

def game(text):
    start = text.index('def terminal_value(')
    return text[:start] + '''DRAW_RULE = "az_actual_threefold_50move_v1"


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
'''

def mcts(text):
    start = text.index('        if board.can_claim_draw():')
    end = text.index('        return value', start)
    text = text[:start] + '        node.children = {move: Node(prior) for move, prior in priors.items()}\n' + text[end:]
    text = text.replace('# None represents the optional action of claiming a draw.', '# Only legal chess moves; terminal draws are checked in expand().')
    text = text.replace('dict[chess.Move | None, Node]', 'dict[chess.Move, Node]')
    text = text.replace('tuple[chess.Move | None, Node]', 'tuple[chess.Move, Node]')
    text = text.replace('            claimed = False\n', '')
    text = text.replace('                if move is None:\n                    claimed = True\n                    break\n', '')
    text = text.replace('            value = 0.0 if claimed else self.expand(node, simulation_board)', '            value = self.expand(node, simulation_board)')
    start = text.index('    def decision_and_target(')
    end = text.index('    def policy(', start)
    text = text[:start] + '''    def decision_and_target(self, board: chess.Board, temperature: float = 1.0):
        moves, probabilities, indices, target = self.policy_and_target(board, temperature)
        selected = moves[int(np.random.choice(len(moves), p=probabilities))]
        return selected, indices, target

    def choose_action(self, board: chess.Board, temperature: float = 0.0) -> chess.Move:
        action, _, _ = self.decision_and_target(board, temperature)
        return action

''' + text[end:]
    text = text.replace('        root = self.search(board)\n        moves =', '        if terminal_value(board) is not None:\n            raise ValueError("Cannot select a move in a terminal position")\n        root = self.search(board)\n        moves =')
    return text

def selfplay(text):
    text = text.replace('from azchess.game import encode_board', 'from azchess.game import encode_board, game_outcome, DRAW_RULE')
    text = text.replace('    args = parser.parse_args()', '    parser.add_argument("--late-temperature", type=float, default=0.25)\n    parser.add_argument("--seed", type=int)\n    args = parser.parse_args()\n    if args.seed is not None:\n        np.random.seed(args.seed)')
    text = text.replace('        claimed_draw = False\n', '')
    text = text.replace('not board.is_game_over(claim_draw=False)', 'game_outcome(board) is None')
    text = text.replace('else 0.25', 'else args.late_temperature')
    text = text.replace('            if move is None:\n                claimed_draw = True\n                break\n', '')
    text = text.replace('board.outcome(claim_draw=claimed_draw)', 'game_outcome(board)')
    text = text.replace('claimed_draw=np.asarray(claimed_draw)', 'claimed_draw=np.asarray(False)')
    text = text.replace('draw_claim_rule=np.asarray("optional_mcts_v1")', 'draw_claim_rule=np.asarray(DRAW_RULE),\n            late_temperature=np.asarray(args.late_temperature)')
    text = text.replace('board.result(claim_draw=claimed_draw)', 'outcome.result()')
    return text

def arena(text):
    text = text.replace('from azchess.mcts import MCTS', 'from azchess.mcts import MCTS\nfrom azchess.game import game_outcome, DRAW_RULE')
    text = text.replace('"optional_mcts_v1"', 'DRAW_RULE')
    text = text.replace('            claimed_draw = False\n', '')
    text = text.replace('not board.is_game_over(claim_draw=False)', 'game_outcome(board) is None')
    text = text.replace('                if move is None:\n                    claimed_draw = True\n                    break\n', '')
    text = text.replace('board.outcome(claim_draw=claimed_draw)', 'game_outcome(board)')
    text = text.replace('board.result(claim_draw=claimed_draw)', 'outcome.result()')
    return text

def train(text):
    text = text.replace('def create_optimizer():', 'def create_optimizer(learning_rate: float = 1e-3):')
    text = text.replace('init_value=1e-3', 'init_value=learning_rate')
    text = text.replace('    args = parser.parse_args()', '    parser.add_argument("--learning-rate", type=float, default=1e-3)\n    parser.add_argument("--seed", type=int)\n    args = parser.parse_args()\n    if args.learning_rate <= 0:\n        parser.error("--learning-rate must be positive")')
    text = text.replace('optimizer = create_optimizer()', 'optimizer = create_optimizer(args.learning_rate)')
    text = text.replace('ReplayBuffer(files, max(1, args.cache_games))', 'ReplayBuffer(files, max(1, args.cache_games), seed=args.seed)')
    return text

for name, transform in [('azchess/game.py', game), ('azchess/mcts.py', mcts), ('selfplay.py', selfplay), ('arena.py', arena), ('train.py', train)]:
    edit(name, transform)
print('Prepared five files; originals preserved in .pre-azrules-20261002')
