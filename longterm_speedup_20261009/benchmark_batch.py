import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import sys,json,time,importlib.util,importlib
import argparse
from pathlib import Path
from collections import deque
import numpy as np,chess,chess.pgn,jax
work=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--simulations',type=int,default=200);parser.add_argument('--workers',default='1,2,4,8');parser.add_argument('--repeats',type=int,default=2)
args=parser.parse_args();sims=args.simulations
for name in ['baseline','candidate']:
    spec=importlib.util.spec_from_file_location(name,work/name/'azchess/__init__.py',submodule_search_locations=[str(work/name/'azchess')])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
old=importlib.import_module('baseline.mcts');new=importlib.import_module('candidate.mcts');batch=importlib.import_module('candidate.batched')
project=Path('/home/user/chess/alphazero_chess_v2')
model,variables=importlib.import_module('candidate.checkpoint').load_variables(project/'experiments/search-budget-20261004/B200/checkpoints/saved/cycle-000050.msgpack')
boards=[]
with (project/'arena_results/match-20261007-085457-07445777.pgn').open() as f:
    for i in range(8):
        game=chess.pgn.read_game(f);board=game.board()
        for move in list(game.mainline_moves())[:-5]:board.push(move)
        boards.append(board)
a=old.MCTS(model,variables,simulations=sims,root_noise=True);a.evaluate(boards[0])
start=time.perf_counter();expected=[]
for i,b in enumerate(boards):
    np.random.seed(90100+i);expected.append(a.search(b))
old_seconds=time.perf_counter()-start
print('BASELINE',old_seconds,flush=True)
results=[]
for workers in map(int,args.workers.split(',')):
    a=new.MCTS(model,variables,simulations=sims,root_noise=True);predictor=batch.BatchedPredictor(a,workers);predictor.warmup()
    times=[];differences=[];max_value_difference=0.;max_prior_difference=0.;total_variations=[];selected_equal=[]
    for repeat in range(args.repeats):
        pending=deque(range(8));active=[];roots={};start=time.perf_counter()
        while pending or active:
            while pending and len(active)<workers:
                i=pending.popleft();coroutine=batch.search_requests(a,boards[i],np.random.RandomState(90100+i))
                active.append((i,coroutine,next(coroutine)))
            output=predictor.predict([entry[2] for entry in active]);left=[]
            for (i,coroutine,_),response in zip(active,output):
                try:left.append((i,coroutine,coroutine.send(response)))
                except StopIteration as done:roots[i]=done.value
            active=left
        times.append(time.perf_counter()-start)
        for i,root in roots.items():
            assert list(root.children)==list(expected[i].children)
            total_variations.append(sum(abs(child.visit_count-expected[i].children[move].visit_count) for move,child in root.children.items())/(2*sims))
            selected_equal.append(max(root.children,key=lambda m:root.children[m].visit_count)==max(expected[i].children,key=lambda m:expected[i].children[m].visit_count))
            for move,child in root.children.items():
                reference=expected[i].children[move]
                if child.visit_count!=reference.visit_count:differences.append(dict(game=i,move=move.uci(),old=reference.visit_count,new=child.visit_count))
                max_prior_difference=max(max_prior_difference,abs(child.prior-reference.prior))
                assert np.isfinite(child.prior) and np.isfinite(child.value)
                # Mean values remain comparable even if tiny batch-rounding
                # differences change visit allocation near a tie.
                if child.visit_count==reference.visit_count:
                    max_value_difference=max(max_value_difference,abs(child.value-reference.value))
    row=dict(workers=workers,seconds=times,speedup=old_seconds/float(np.median(times)),visit_differences=differences,
        exact_visit_counts=not differences,max_value_difference=max_value_difference,
        max_prior_difference=max_prior_difference,max_policy_total_variation=max(total_variations),selected_move_equal=all(selected_equal))
    results.append(row);print('BATCH',json.dumps(row),flush=True)
out=dict(device=jax.default_backend(),positions=8,simulations=sims,baseline_seconds=old_seconds,results=results)
(work/('batch_800_validation.json' if sims==800 else 'batch_validation.json')).write_text(json.dumps(out,indent=2));print('DONE',flush=True)
