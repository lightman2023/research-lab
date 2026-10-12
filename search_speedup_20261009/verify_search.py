import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import sys, importlib, importlib.util, json, time, cProfile, pstats, io
from pathlib import Path
import numpy as np, chess, chess.pgn, jax

work=Path(__file__).resolve().parent
project=Path('/home/user/chess/alphazero_chess_v2')
for label in ('baseline','candidate'):
    spec=importlib.util.spec_from_file_location(label,work/label/'azchess/__init__.py',submodule_search_locations=[str(work/label/'azchess')])
    module=importlib.util.module_from_spec(spec);sys.modules[label]=module;spec.loader.exec_module(module)
model,variables=importlib.import_module('candidate.checkpoint').load_variables(project/'experiments/search-budget-20261004/B200/checkpoints/saved/cycle-000050.msgpack')
boards=[]
with (project/'arena_results/match-20261007-085457-07445777.pgn').open() as f:
    for i in range(8):
        game=chess.pgn.read_game(f);board=game.board()
        for move in list(game.mainline_moves())[:-5]:board.push(move)
        boards.append(board)
boards[0]=chess.Board()
boards[1]=chess.Board()
for move in ('e2e4','e7e5','g1f3','b8c6','f1b5'):boards[1].push_uci(move)

def run(label,sims,profile=False):
    mcts=importlib.import_module(label+'.mcts');batch=importlib.import_module(label+'.batched')
    agent=mcts.MCTS(model,variables,simulations=sims,root_noise=True)
    predictor=batch.BatchedPredictor(agent,8);predictor.warmup()
    coroutines=[batch.search_requests(agent,b,np.random.RandomState(90100+i)) for i,b in enumerate(boards)]
    active=[(i,c,next(c)) for i,c in enumerate(coroutines)];roots={}
    timer=cProfile.Profile() if profile else None
    if timer:timer.enable()
    start=time.perf_counter()
    while active:
        responses=predictor.predict([entry[2] for entry in active]);left=[]
        for (i,c,_),response in zip(active,responses):
            try:left.append((i,c,c.send(response)))
            except StopIteration as done:roots[i]=done.value
        active=left
    seconds=time.perf_counter()-start
    if timer:
        timer.disable();stream=io.StringIO()
        pstats.Stats(timer,stream=stream).sort_stats('cumtime').print_stats(25)
        (work/(label+'_profile.txt')).write_text(stream.getvalue())
    return roots,seconds,predictor.calls

rows=[]
for repeat in range(3):
    results={}
    # Alternate order to reduce systematic warm-up/temperature bias.
    for label in (('baseline','candidate') if repeat%2==0 else ('candidate','baseline')):
        results[label]=run(label,800)
    old,oldtime,oldcalls=results['baseline'];new,newtime,newcalls=results['candidate']
    compared=0
    def compare(a,b):
        global compared
        assert list(a.children)==list(b.children)
        assert (a.visit_count,a.value_sum,a.prior)==(b.visit_count,b.value_sum,b.prior)
        compared+=1
        for move in a.children:compare(a.children[move],b.children[move])
    for i in old:compare(old[i],new[i])
    assert oldcalls==newcalls
    rows.append(dict(repeat=repeat,baseline_seconds=oldtime,candidate_seconds=newtime,speedup=oldtime/newtime,exact_tree_nodes=compared,inference_calls=oldcalls))
    print(json.dumps(rows[-1]),flush=True)
for label in ('baseline','candidate'):run(label,200,profile=True)
out=dict(device=jax.default_backend(),simulations=800,workers=8,positions=8,seed_base=90100,checkpoint='saved/cycle-000050.msgpack',results=rows)
(work/'search_validation.json').write_text(json.dumps(out,indent=2))
print('DONE',flush=True)
