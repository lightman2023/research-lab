import sys
import json
import subprocess
from pathlib import Path
sys.path.insert(0,'/home/user/chess/alphazero_chess_v2')
import chess.pgn

project=Path('/home/user/chess/alphazero_chess_v2')
model=project/'experiments/live-lr0001-temp1-20261002/checkpoints/saved/cycle-000010.msgpack'
command=[sys.executable,'-u',str(project/'arena.py'),'--model-a',str(model),
    '--model-b',str(model),'--games','2','--simulations-a','50',
    '--simulations-b','200','--max-plies','20','--match-id','search-smoke-20261003']
run=subprocess.run(command,cwd=project,capture_output=True,text=True)
Path(__file__).with_name('search_comparison_smoke.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stderr
events=[json.loads(line) for line in run.stdout.splitlines()]
summary=events[-1]
assert summary['event']=='summary' and summary['games']==2
assert summary['simulations_a']==50 and summary['simulations_b']==200
games=[]
with open(summary['pgn']) as f:
    while (game:=chess.pgn.read_game(f)) is not None:games.append(game)
assert len(games)==2
for index,game in enumerate(games):
    assert not game.errors
    assert game.headers['WhiteSimulations']==str(50 if index==0 else 200)
    assert game.headers['BlackSimulations']==str(200 if index==0 else 50)
    assert game.headers['White'].startswith('A:' if index==0 else 'B:')
metadata=json.loads(Path(summary['metadata']).read_text())
assert metadata['model_a_sha256']==metadata['model_b_sha256']
assert len(metadata['results'])==len(games)==2
assert metadata['results'][0]['opening']==metadata['results'][1]['opening']
result=dict(smoke_games=2,simulations_a=50,simulations_b=200,max_plies=20,
    colors_swapped_correctly=True,same_model_hash=True,pgn_count_matches=True,
    pgn=summary['pgn'],metadata=summary['metadata'],not_strength_evaluation=True)
Path(__file__).with_name('search_comparison_smoke_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
