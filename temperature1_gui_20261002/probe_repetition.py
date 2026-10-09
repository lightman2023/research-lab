import sys
import json
import hashlib
from pathlib import Path
from collections import Counter

PROJECT = Path('/home/user/chess/alphazero_chess_v2')
sys.path.insert(0, str(PROJECT))
import chess
import chess.pgn
from azchess.mcts import MCTS
from azchess.checkpoint import load_variables
from azchess.game import game_outcome

PGN = PROJECT / 'arena_results/match-20261002-233038-c11626a7.pgn'
OUT = Path(__file__).with_name('repetition_probe_results.json')

class Probe(MCTS):
    def search(self, board):
        self.root_ply = len(board.move_stack)
        self.leaves = Counter()
        self.draws = Counter()
        return super().search(board)

    def expand(self, node, board):
        if len(board.move_stack) > self.root_ply:
            first = board.move_stack[self.root_ply].uci()
            self.leaves[first] += 1
            outcome = game_outcome(board)
            if outcome and outcome.winner is None:
                self.draws[first] += 1
        return super().expand(node, board)

games = []
with PGN.open() as f:
    while (game := chess.pgn.read_game(f)) is not None:
        assert not game.errors
        games.append(game)
assert len(games) == 17, len(games)
assert all(game_outcome(g.end().board()).termination == chess.Termination.THREEFOLD_REPETITION for g in games)
agents = {}
hashes = {}
for cycle in (0, 10):
    path = PROJECT / f'experiments/live-lr0001-temp1-20261002/checkpoints/saved/cycle-{cycle:06d}.msgpack'
    model, variables = load_variables(path)
    agents[cycle] = Probe(model, variables)
    hashes[str(cycle)] = hashlib.sha256(path.read_bytes()).hexdigest()

records = []
for number in (1, 2, 3):
    game = games[number - 1]
    moves = list(game.mainline_moves())
    for offset in (5, 1):
        ply = len(moves) - offset
        board = game.board()
        for move in moves[:ply]:
            board.push(move)
        assert game_outcome(board) is None
        for cycle, agent in agents.items():
            for sims in ((50, 200) if offset == 1 else (50,)):
                agent.simulations = sims
                _, value = agent.evaluate(board)
                root = agent.search(board)
                assert sum(c.visit_count for c in root.children.values()) == sims
                rows = []
                for move, child in root.children.items():
                    successor = board.copy(stack=True)
                    successor.push(move)
                    outcome = game_outcome(successor)
                    rows.append(dict(uci=move.uci(), san=board.san(move),
                        visits=child.visit_count, prior=child.prior,
                        q=-child.value if child.visit_count else None,
                        immediate_termination=outcome.termination.name if outcome else None,
                        terminal_draw_leaves=agent.draws[move.uci()],
                        reverse_own_last_move=(ply >= 2 and move.from_square == moves[ply-2].to_square and move.to_square == moves[ply-2].from_square)))
                rows.sort(key=lambda r: -r['visits'])
                records.append(dict(game=number, ply=ply, offset_to_end=offset,
                    model_cycle=cycle, simulations=sims, root_value=value,
                    actual_move=moves[ply].uci(), fen=board.fen(),
                    is_second_occurrence=board.is_repetition(2), moves=rows))
                print(json.dumps(dict(game=number, ply=ply, cycle=cycle, sims=sims,
                    v=round(value,4), selected=rows[0], actual=moves[ply].uci()), ensure_ascii=True), flush=True)
result = dict(pgn=str(PGN), pgn_sha256=hashlib.sha256(PGN.read_bytes()).hexdigest(),
    completed_games=len(games), terminations={'THREEFOLD_REPETITION':len(games)},
    model_sha256=hashes, root_noise=False, history_preserved=True,
    q_perspective='player to move at root; unvisited Q unknown',
    position_selection='first three completed games, five and one plies before end', records=records)
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print('SAVED ' + str(OUT), flush=True)
