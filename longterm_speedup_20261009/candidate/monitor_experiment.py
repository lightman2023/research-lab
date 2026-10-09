import argparse
import json
import os
from pathlib import Path
from collections import Counter
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import chess
from flax import linen as nn, serialization
import jax
import jax.numpy as jnp
import numpy as np
from azchess.checkpoint import load_variables
from azchess.game import encode_board
from train import ReplayBuffer
from azchess.data_cache import DataCache

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--cycle',type=int,required=True)
    args=parser.parse_args(); root=args.experiment
    manifest=json.loads((root/'evaluation_manifest.json').read_text())
    states=[]; labels=[]; groups=[];fixed_games={}
    for item in manifest:
        if 'moves' in item:
            board=chess.Board()
            for uci in item['moves']: board.push_uci(uci)
            states.append(encode_board(board)); labels.append(0.)
        else:
            if item['file'] not in fixed_games:
                with np.load(item['file']) as game:
                    fixed_games[item['file']]=(game['states'],game['values'])
            game_states,game_values=fixed_games[item['file']]
            states.append(game_states[item['position']].astype(np.float32))
            labels.append(float(game_values[item['position']]))
        groups.append(item['group'])
    del fixed_games
    model,variables=load_variables(root/'checkpoints/latest.msgpack')
    @jax.jit
    def inspect(variables,x):
        (_,value),extra=model.apply(variables,x,train=False,capture_intermediates=lambda mod,method:isinstance(mod,nn.Dense),mutable=['intermediates'])
        return value,extra['intermediates']['Dense_0']['__call__'][0]
    def evaluate(x,z,groups):
        values,dense=map(np.asarray,inspect(variables,jnp.asarray(x)))
        assert np.isfinite(values).all() and np.isfinite(dense).all()
        dead=np.all(dense<=0,axis=1); groups=np.asarray(groups); z=np.asarray(z)
        result=dict(positions=len(values),dead=int(dead.sum()),value_std=float(values.std()),groups={})
        mask=groups!='opening'
        result['value_mse']=float(np.mean((values[mask]-z[mask])**2))
        for group in sorted(set(groups)):
            m=groups==group
            result['groups'][str(group)]=dict(positions=int(m.sum()),dead=int(dead[m].sum()),value_mean=float(values[m].mean()))
            if group!='opening': result['groups'][str(group)]['value_mse']=float(np.mean((values[m]-z[m])**2))
        return result
    fixed=evaluate(np.stack(states),labels,groups)
    files=sorted((root/'selfplay_data').glob('*.npz'))[-1000:]
    replay=ReplayBuffer(files,64,seed=20261003+args.cycle)
    x,p,z=replay.sample(128)
    live=evaluate(x,z,[str(int(v)) for v in z])
    outcomes=Counter(); label_counts=Counter(); plies=[]
    cycle_outcomes=Counter(); new_games=[]
    metadata=DataCache(root/'selfplay_data')
    for path in files:
        info=metadata.get(path);outcomes[info['termination']]+=1;plies.append(info['positions']);label_counts.update(info['label_counts'])
        if path.name.startswith('game-') and info['cycle']==args.cycle:
            cycle_outcomes[info['termination']]+=1;new_games.append(str(path))
    metadata.flush()
    step=int(serialization.msgpack_restore((root/'checkpoints/training_state.msgpack').read_bytes())['step'])
    stop=fixed['dead']/fixed['positions']>=.1 or live['dead']/live['positions']>=.1
    result=dict(cycle=args.cycle,training_step=step,fixed=fixed,live_training_sample=live,
                replay_games=len(files),replay_outcomes=dict(outcomes),labels=dict(label_counts),
                replay_mean_plies=float(np.mean(plies)),cycle_outcomes=dict(cycle_outcomes),cycle_games=new_games,
                stop_due_to_widespread_inactivity=stop,stop_threshold_fraction=.1)
    (root/'metrics').mkdir(exist_ok=True)
    path=root/'metrics'/f'cycle-{args.cycle:06d}.json'
    if path.exists(): raise RuntimeError(f'Metrics already exist: {path}')
    path.write_text(json.dumps(result,indent=2))
    print('MONITOR '+json.dumps(result),flush=True)
    return 2 if stop else 0

if __name__=='__main__': raise SystemExit(main())
