import sys
import json
import hashlib
from pathlib import Path
from collections import Counter
from statistics import mean,median

project=Path('/home/user/chess/alphazero_chess_v2')
sys.path.insert(0,str(project))
import chess.pgn
from azchess.game import game_outcome

pgn=project/'arena_results/match-20261003-002452-fbce438a.pgn'
meta=json.loads(pgn.with_suffix('.json').read_text())
games=[]
with pgn.open() as f:
    while (game:=chess.pgn.read_game(f)) is not None:games.append(game)
assert len(games)==len(meta['results'])==meta['games_requested']==24
assert meta['simulations_a']==50 and meta['simulations_b']==200
assert meta['model_a']==meta['model_b']
assert meta['model_a_sha256']==meta['model_b_sha256']==hashlib.sha256(Path(meta['model_a']).read_bytes()).hexdigest()
rows=[]
for index,(game,entry) in enumerate(zip(games,meta['results'])):
    assert not game.errors
    board=game.board()
    for move in game.mainline_moves():
        assert game_outcome(board) is None, ('continued past termination',index)
        assert move in board.legal_moves
        board.push(move)
    outcome=game_outcome(board)
    assert outcome and outcome.result()==entry['result']==game.headers['Result']
    assert outcome.termination.name==entry['termination']==game.headers['Termination']
    assert len(board.move_stack)==entry['plies']
    assert game.headers['WhiteSimulations']==str(50 if index%2==0 else 200)
    assert game.headers['BlackSimulations']==str(200 if index%2==0 else 50)
    if index%2:assert game.headers['Opening']==games[index-1].headers['Opening']
    rows.append(entry)
counts=Counter(entry['winner'] for entry in rows)
reasons=Counter(entry['termination'] for entry in rows)
result=dict(games=24,same_model_verified=True,simulations_a=50,simulations_b=200,
    max_plies=meta['max_plies'],seed=meta['seed'],a_wins=counts['A'],b_wins=counts['B'],
    draws=counts['draw'],terminations=dict(reasons),b_score_rate=(counts['B']+.5*counts['draw'])/24,
    mean_plies=mean(r['plies'] for r in rows),median_plies=median(r['plies'] for r in rows),
    plies_range=[min(r['plies'] for r in rows),max(r['plies'] for r in rows)],
    pgn_legality_and_earliest_termination_verified=True,color_swap_verified=True,
    decisive_games=[r for r in rows if r['winner']!='draw'],
    shortest_game_pgn=str(min(games,key=lambda g:len(list(g.mainline_moves())))),
    model_sha256=meta['model_a_sha256'],pgn=str(pgn))
Path(__file__).with_name('search_match_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=True,indent=2))
