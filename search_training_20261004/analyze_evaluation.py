import json
import sys
import hashlib
from pathlib import Path
from collections import Counter
from statistics import mean

project=Path('/home/user/chess/alphazero_chess_v2');sys.path.insert(0,str(project))
import chess
import chess.pgn
from azchess.game import game_outcome
pgn=project/'arena_results/match-20261005-090556-8608c126.pgn'
metadata=json.loads(pgn.with_suffix('.json').read_text())
assert metadata['simulations_a']==metadata['simulations_b']==200
for side,branch in [('a','A50'),('b','B200')]:
    expected=project/f'experiments/search-budget-20261004/{branch}/checkpoints/saved/cycle-000005.msgpack'
    assert Path(metadata[f'model_{side}']).resolve()==expected.resolve()
    assert metadata[f'model_{side}_sha256']==hashlib.sha256(expected.read_bytes()).hexdigest()
games=[]
with pgn.open() as f:
    while (game:=chess.pgn.read_game(f)) is not None:games.append(game)
assert len(games)==len(metadata['results'])==metadata['games_requested']==12
for index,(game,entry) in enumerate(zip(games,metadata['results'])):
    assert not game.errors
    board=game.board()
    for move in game.mainline_moves():
        assert game_outcome(board) is None
        assert move in board.legal_moves
        board.push(move)
    outcome=game_outcome(board)
    assert outcome and outcome.result()==entry['result']==game.headers['Result']
    assert outcome.termination.name==entry['termination']==game.headers['Termination']
    assert len(board.move_stack)==entry['plies']
    assert game.headers['WhiteSimulations']==game.headers['BlackSimulations']=='200'
    assert game.headers['White'].startswith('A:' if index%2==0 else 'B:')
    if index%2:assert game.headers['Opening']==games[index-1].headers['Opening']
counts=Counter(row['winner'] for row in metadata['results'])
result=dict(games=12,openings=6,simulations_both=200,a_wins=counts['A'],b_wins=counts['B'],draws=counts['draw'],
    terminations=dict(Counter(row['termination'] for row in metadata['results'])),
    mean_plies=mean(row['plies'] for row in metadata['results']),
    b_score_rate=(counts['B']+.5*counts['draw'])/12,
    wins_by_opening=[row for row in metadata['results'] if row['winner']!='draw'],
    model_hashes_verified=True,pgn_count_and_legality_verified=True,color_swap_verified=True,
    pgn=str(pgn),conclusion='Inconclusive strength difference; repetition remains; improvement over starting model untested.')
Path(__file__).with_name('evaluation_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=True))
