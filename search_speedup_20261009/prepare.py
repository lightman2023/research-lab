from pathlib import Path
import shutil

work = Path(__file__).resolve().parent
project = Path('/home/user/chess/alphazero_chess_v2')
for label in ('baseline', 'candidate'):
    shutil.copytree(project/'azchess', work/label/'azchess', dirs_exist_ok=True)

path = work/'candidate/azchess/history.py'
s = path.read_text()
s = s.replace('self.board=board', 'self.board=board\n        self._piece_cache={}')
s = s.replace('self.frames.append((piece_planes(self.board),self.counts[key]))', '''# The first eight key entries describe piece placement and ownership.
        # Revisited search paths can share immutable planes; repetition counts
        # remain path-specific and are never cached with the piece placement.
        piece_key=key[:8]
        pieces=self._piece_cache.get(piece_key)
        if pieces is None:
            pieces=piece_planes(self.board)
            if len(self._piece_cache)<2048:self._piece_cache[piece_key]=pieces
        self.frames.append((pieces,self.counts[key]))''')
start = s.index('    def outcome(self):')
s = s[:start] + '''    def outcome(self, legal_moves=None):
        board=self.board
        has_moves=any(board.generate_legal_moves()) if legal_moves is None else bool(legal_moves)
        if not has_moves and board.is_check():
            return chess.Outcome(chess.Termination.CHECKMATE,not board.turn)
        if board.is_insufficient_material():return chess.Outcome(chess.Termination.INSUFFICIENT_MATERIAL,None)
        if not has_moves:return chess.Outcome(chess.Termination.STALEMATE,None)
        # Both move-count rules require a legal move; already established above.
        if board.halfmove_clock>=150:return chess.Outcome(chess.Termination.SEVENTYFIVE_MOVES,None)
        count=self.counts[self.keys[-1]]
        if count>=5:return chess.Outcome(chess.Termination.FIVEFOLD_REPETITION,None)
        if count>=3:return chess.Outcome(chess.Termination.THREEFOLD_REPETITION,None)
        if board.halfmove_clock>=100:return chess.Outcome(chess.Termination.FIFTY_MOVES,None)
        return None

    def terminal_value(self, legal_moves=None):
        outcome=self.outcome(legal_moves)
        if outcome is None:return None
        if outcome.winner is None:return 0.
        return 1. if outcome.winner==self.board.turn else -1.
'''
path.write_text(s)
path = work/'candidate/azchess/game.py'
s = path.read_text().replace('def legal_action_map(board: chess.Board)', 'def legal_action_map(board: chess.Board, moves=None)').replace('    moves = list(board.legal_moves)\n', '    if moves is None: moves = list(board.legal_moves)\n')
path.write_text(s)
path = work/'candidate/azchess/mcts.py'
s = path.read_text().replace('class Node:\n', 'class Node:\n    __slots__ = ("prior", "visit_count", "value_sum", "children", "terminal_value")\n\n').replace('        self.prior = prior', '        self.terminal_value = None\n        self.prior = prior')
s = s.replace('def evaluate(self, board: chess.Board)', 'def evaluate(self, board: chess.Board, action_map=None)').replace('        action_map = legal_action_map(board)', '        if action_map is None: action_map = legal_action_map(board)')
start=s.index('        terminal = self._encoder.terminal_value()')
end=s.index('        node.children =',start)
s=s[:start]+'''        if node.terminal_value is not None:
            return node.terminal_value
        incremental = self._encoder is not None and self._encoder.board is board
        moves = list(board.legal_moves) if incremental else None
        terminal = self._encoder.terminal_value(moves) if incremental else terminal_value(board)
        if terminal is not None:
            node.terminal_value = terminal
            return terminal
        priors, value = self.evaluate(board, legal_action_map(board, moves))
'''+s[end:]
start=s.index('        return max(\n',s.index('    def select_child'))
end=s.index('\n    def search(',start)
s=s[:start]+'''        best_score = -math.inf
        best_move = best_child = None
        for move, child in node.children.items():
            count = child.visit_count
            value = child.value_sum / count if count else 0.0
            score = -value + child.prior * scale / (1 + count)
            if score > best_score:
                best_score = score
                best_move, best_child = move, child
        return best_move, best_child
'''+s[end:]
path.write_text(s)
path=work/'candidate/azchess/batched.py'
s=path.read_text().replace('        terminal=encoder.terminal_value()', '''        if node.terminal_value is not None:return node.terminal_value
        moves=list(working.legal_moves)
        terminal=encoder.terminal_value(moves)''').replace('        if terminal is not None:return terminal', '''        if terminal is not None:
            node.terminal_value=terminal
            return terminal''').replace('legal_action_map(working)', 'legal_action_map(working,moves)')
path.write_text(s)
print('Prepared candidate in', work)
