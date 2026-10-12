from __future__ import annotations
import argparse,json,signal,time
from collections import deque
from datetime import datetime
from pathlib import Path
import chess
import numpy as np
from azchess.checkpoint import load_variables
from azchess.mcts import MCTS
from azchess.batched import BatchedPredictor,search_requests
from azchess.history import HistoryEncoder
from azchess.game import DRAW_RULE
from azchess.data_cache import DataCache

def play(agent,args,number,rng):
    board=chess.Board();encoder=HistoryEncoder(board)
    states=[];indices=[];policies=[];offsets=[0];players=[];moves=[]
    while encoder.outcome() is None and len(moves)<args.max_plies:
        root=yield from search_requests(agent,board,rng)
        temperature=1. if len(moves)<30 else args.late_temperature
        legal,p,action_indices,target=agent.policy_from_root(board,root,temperature)
        move=legal[int(rng.choice(len(legal),p=p))]
        states.append(encoder.encode().astype(np.float16));indices.append(action_indices)
        policies.append(target.astype(np.float16));offsets.append(offsets[-1]+len(action_indices))
        players.append(board.turn);moves.append(move.uci());board.push(move);encoder.push()
        if len(moves)%10==0:print(f'Game {number} progress: plies={len(moves)} last={move.uci()}',flush=True)
    outcome=encoder.outcome();winner=outcome.winner if outcome else None
    termination=outcome.termination.name if outcome else 'MAX_PLIES_DRAW'
    values=np.asarray([0. if winner is None else (1. if winner==player else -1.) for player in players],dtype=np.float32)
    path=args.output_dir/f'game-{datetime.now():%Y%m%d-%H%M%S-%f}.npz'
    temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,states=np.stack(states),policy_offsets=np.asarray(offsets,dtype=np.int32),
            policy_indices=np.concatenate(indices).astype(np.uint16),policy_values=np.concatenate(policies).astype(np.float16),
            values=values,moves=np.asarray(moves,dtype='<U5'),final_fen=np.asarray(board.fen()),termination=np.asarray(termination),
            claimed_draw=np.asarray(False),draw_claim_rule=np.asarray(DRAW_RULE),late_temperature=np.asarray(args.late_temperature),
            experiment_cycle=np.asarray(args.experiment_cycle),seed=np.asarray(-1 if args.seed is None else args.seed+number),
            policy_target_kind=np.asarray('raw_mcts_visits_v1'),simulations=np.asarray(args.simulations),search_engine=np.asarray('batched_sequential_v1'),
            inference_precision=np.asarray('float32_default' if args.workers==1 else 'float32_highest'))
    temporary.replace(path)
    result=outcome.result() if outcome else '1/2-1/2'
    print(f'Game {number}: {result} termination={termination} plies={len(moves)} -> {path}',flush=True)
    return path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--games',type=int,default=8);parser.add_argument('--game-numbers')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--simulations',type=int,default=800)
    parser.add_argument('--max-plies',type=int,default=512);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--late-temperature',type=float,default=1.)
    parser.add_argument('--seed',type=int);parser.add_argument('--experiment-cycle',type=int,default=-1)
    args=parser.parse_args()
    numbers=list(range(1,args.games+1)) if args.game_numbers is None else [int(n) for n in args.game_numbers.split(',')]
    if not 1<=args.workers<=8 or args.max_plies<1 or args.simulations<1:parser.error('Invalid workers/plies/simulations')
    if len(set(numbers))!=len(numbers) or any(n<1 or n>args.games for n in numbers):parser.error('Invalid game numbers')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # A retry resumes completed numbered games instead of duplicating samples.
    completed=set()
    if args.seed is not None:
        metadata=DataCache(args.output_dir)
        for path in args.output_dir.glob('game-*.npz'):
            info=metadata.get(path)
            if info['cycle']==args.experiment_cycle:
                n=info['seed']-args.seed
                if n in completed:raise ValueError('Duplicate completed game seed')
                completed.add(n)
        metadata.flush()
    pending=deque(n for n in numbers if n not in completed)
    if not pending:print('All requested games already saved',flush=True);return 0
    model,variables=load_variables(args.checkpoint)
    agent=MCTS(model,variables,simulations=args.simulations,root_noise=True)
    predictor=BatchedPredictor(agent,args.workers);predictor.warmup()
    stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    active=[];start=time.perf_counter()
    print(f'Batched self-play: {args.workers} games at once, {args.simulations} simulations, one shared model',flush=True)
    while (pending or active) and not stopping:
        while pending and len(active)<args.workers:
            n=pending.popleft();rng=np.random.RandomState(None if args.seed is None else args.seed+n)
            coroutine=play(agent,args,n,rng)
            request=next(coroutine);active.append((n,coroutine,request))
            print(f'Parallel game {n}/{args.games} started',flush=True)
        responses=predictor.predict([entry[2] for entry in active]);remaining=[]
        for (n,coroutine,_),response in zip(active,responses):
            try:remaining.append((n,coroutine,coroutine.send(response)))
            except StopIteration:print(f'Parallel game {n}/{args.games} finished',flush=True)
        active=remaining
    if stopping:
        print('Stopped; completed games preserved',flush=True);return 1
    print('SELFPLAY_METRICS '+json.dumps(dict(seconds=time.perf_counter()-start,
        inference_calls=predictor.calls,evaluated_positions=predictor.positions,batch_histogram=dict(predictor.batch_histogram),workers=args.workers,simulations=args.simulations)),flush=True)
    print(f'All requested {len(numbers)} parallel self-play games finished',flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
