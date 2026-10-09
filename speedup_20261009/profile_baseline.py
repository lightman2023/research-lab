import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,time,cProfile,pstats,io,json
from pathlib import Path
work=Path(__file__).resolve().parent
sys.path.insert(0,str(work/'baseline'))
import chess,chess.pgn,jax
from azchess.checkpoint import load_variables
from azchess.mcts import MCTS
project=Path('/home/user/chess/alphazero_chess_v2')
with (project/'arena_results/match-20261007-085457-07445777.pgn').open() as f:
    games=[]
    while (g:=chess.pgn.read_game(f)) is not None:games.append(g)
g=games[21];b=g.board()
for m in list(g.mainline_moves())[:-5]:b.push(m)
model,v=load_variables(project/'experiments/search-budget-20261004/B200/checkpoints/saved/cycle-000050.msgpack')
agent=MCTS(model,v,simulations=64)
agent.evaluate(b)
profile=cProfile.Profile();profile.enable();start=time.perf_counter();agent.search(b);seconds=time.perf_counter()-start;profile.disable()
stream=io.StringIO();pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(20)
(work/'profile_baseline.txt').write_text(stream.getvalue())
print(json.dumps(dict(seconds=seconds,plies=len(b.move_stack),variable_types=dict(__import__('collections').Counter(type(x).__name__ for x in jax.tree_util.tree_leaves(v))),weight_bytes=sum(x.nbytes for x in jax.tree_util.tree_leaves(v)))))
print(stream.getvalue())
