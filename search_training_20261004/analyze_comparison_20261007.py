import json, hashlib, sys
from pathlib import Path
from collections import Counter
from statistics import mean
project=Path('/home/user/chess/alphazero_chess_v2');sys.path.insert(0,str(project))
import chess.pgn
from azchess.game import game_outcome
pgn=project/'arena_results/match-20261007-085457-07445777.pgn'
meta=json.loads(pgn.with_suffix('.json').read_text())
assert meta['simulations_a']==meta['simulations_b']==200 and meta['device']=='cpu'
for side in ['a','b']:
    assert hashlib.sha256(Path(meta['model_'+side]).read_bytes()).hexdigest()==meta['model_'+side+'_sha256']
games=[]
with pgn.open() as stream:
    while (game:=chess.pgn.read_game(stream)) is not None:games.append(game)
assert len(games)==len(meta['results'])==meta['games_requested']==24
for i,(game,row) in enumerate(zip(games,meta['results'])):
    assert not game.errors
    board=game.board()
    for move in game.mainline_moves():
        assert game_outcome(board) is None and move in board.legal_moves
        board.push(move)
    outcome=game_outcome(board)
    assert outcome and outcome.result()==row['result']==game.headers['Result']
    assert outcome.termination.name==row['termination']
    assert len(board.move_stack)==row['plies']
    assert game.headers['White'].startswith('A:' if i%2==0 else 'B:')
    if i%2:assert game.headers['Opening']==games[i-1].headers['Opening']
counts=Counter(row['winner'] for row in meta['results'])
result=dict(model_a=meta['model_a'],model_b=meta['model_b'],games=24,openings=12,
    wins=dict(counts),terminations=dict(Counter(r['termination'] for r in meta['results'])),
    b_score=(counts['B']+.5*counts['draw'])/24,mean_plies=mean(r['plies'] for r in meta['results']),
    a_win_plies=[r['plies'] for r in meta['results'] if r['winner']=='A'],
    hashes_verified=True,pgn_count_legality_terminal_and_color_swap_verified=True,
    pgn=str(pgn),settings={k:meta[k] for k in ['simulations_a','simulations_b','device','seed','max_plies']})
Path(__file__).with_name('comparison_20261007_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False))
