import os,sys
gpu='--gpu' in sys.argv
if gpu:
    os.environ.pop('JAX_PLATFORMS',None)
else:
    os.environ['JAX_PLATFORMS']='cpu'
import sys,importlib.util,importlib,time,json,hashlib
from pathlib import Path
from statistics import median
import numpy as np
import chess,chess.pgn,jax
work=Path(__file__).resolve().parent
project=Path('/home/user/chess/alphazero_chess_v2')
for label in ['baseline','candidate']:
    spec=importlib.util.spec_from_file_location(label,work/label/'azchess/__init__.py',submodule_search_locations=[str(work/label/'azchess')])
    package=importlib.util.module_from_spec(spec);sys.modules[label]=package;spec.loader.exec_module(package)
old=importlib.import_module('baseline.mcts');new=importlib.import_module('candidate.mcts')
game_old=importlib.import_module('baseline.game');game_new=importlib.import_module('candidate.game')
checkpoint=project/'experiments/search-budget-20261004/B200/checkpoints/saved/cycle-000050.msgpack'
model,variables=importlib.import_module('baseline.checkpoint').load_variables(checkpoint)
games=[]
with (project/'arena_results/match-20261007-085457-07445777.pgn').open() as stream:
    while (g:=chess.pgn.read_game(stream)) is not None:games.append(g)
boards=[chess.Board()]
for index,offset in [(5,5),(21,2),(23,1)]:
    b=games[index].board()
    for move in list(games[index].mainline_moves())[:-offset]:b.push(move)
    boards.append(b)
checks=[]
for index,board in enumerate(boards):
    if gpu and index not in [0,2]:continue
    np.testing.assert_array_equal(game_old.encode_board(board),game_new.encode_board(board))
    assert game_old.legal_action_map(board)==game_new.legal_action_map(board)
    before=(board.fen(),tuple(board.move_stack))
    for noise in [False,True]:
        a=old.MCTS(model,variables,simulations=200,root_noise=noise)
        b=new.MCTS(model,variables,simulations=200,root_noise=noise)
        np.random.seed(7200+index);root_a=a.search(board)
        np.random.seed(7200+index);root_b=b.search(board)
        assert list(root_a.children)==list(root_b.children)
        for move,child_a in root_a.children.items():
            child_b=root_b.children[move]
            assert child_a.visit_count==child_b.visit_count
            np.testing.assert_allclose([child_a.prior,child_a.value_sum],[child_b.prior,child_b.value_sum],rtol=0,atol=1e-6)
        assert (board.fen(),tuple(board.move_stack))==before
        checks.append(dict(position=index,plies=len(board.move_stack),noise=noise,visits_equal=True))
    print('VERIFIED',index,len(board.move_stack),flush=True)
timings=[]
for index in ([0,2] if gpu else [0,2,3]):
    board=boards[index]
    simulations=200 if gpu else 800
    agents={'baseline':old.MCTS(model,variables,simulations=simulations),'candidate':new.MCTS(model,variables,simulations=simulations)}
    for a in agents.values():a.evaluate(board)
    results={key:[] for key in agents}
    for repeat in range(3):
        order=['baseline','candidate'] if repeat%2==0 else ['candidate','baseline']
        for key in order:
            start=time.perf_counter();agents[key].search(board);results[key].append(time.perf_counter()-start)
    row=dict(position=index,plies=len(board.move_stack),simulations=simulations,seconds=results,
        speedup=median(results['baseline'])/median(results['candidate']))
    timings.append(row);print('BENCH',json.dumps(row),flush=True)
result=dict(device=jax.default_backend(),checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checks=checks,timings=timings,
    candidate_weight_types=sorted(set(type(x).__name__ for x in jax.tree_util.tree_leaves(agents['candidate'].variables))))
assert jax.default_backend()==('gpu' if gpu else 'cpu')
(work/('search_gpu_validation.json' if gpu else 'search_validation.json')).write_text(json.dumps(result,indent=2))
print('DONE',flush=True)
