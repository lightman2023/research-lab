import json
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from flax import serialization

ROOT=Path('/home/user/chess/alphazero_chess_v2/experiments/search-budget-20261004')
results=[]
for branch in ('A50','B200'):
    root=ROOT/branch
    metrics=[json.loads((root/'metrics'/f'cycle-{cycle:06d}.json').read_text()) for cycle in range(6)]
    paths=[Path(path) for m in metrics[1:] for path in m['cycle_games']]
    assert len(paths)==len(set(paths))==40
    assert all(len(m['cycle_games'])==8 for m in metrics[1:])
    outcomes=Counter();plies=[];seeds=set()
    for p in paths:
        with np.load(p) as game:
            key=(int(game['experiment_cycle']),int(game['seed']))
            assert key not in seeds;seeds.add(key)
            outcomes[str(game['termination'])]+=1;plies.append(len(game['values']))
    latest=root/'checkpoints/latest.msgpack';saved=root/'checkpoints/saved/cycle-000005.msgpack'
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(latest)==sha(saved)
    step=int(serialization.msgpack_restore((root/'checkpoints/training_state.msgpack').read_bytes())['step'])
    assert step==3200 and metrics[-1]['training_step']==3200
    assert len(list((root/'selfplay_data').glob('*.npz')))==200
    results.append(dict(branch=branch,cycles=5,new_games=40,training_step=step,outcomes=dict(outcomes),
        mean_plies=float(np.mean(plies)),fixed_dead=[m['fixed']['dead'] for m in metrics],
        live_dead=[m['live_training_sample']['dead'] for m in metrics],
        fixed_mse_start=metrics[0]['fixed']['value_mse'],fixed_mse_end=metrics[-1]['fixed']['value_mse'],
        repetition_prior_start=[m['reference_prior'] for m in metrics[0]['repetition_policy_probe']],
        repetition_prior_end=[m['reference_prior'] for m in metrics[-1]['repetition_policy_probe']],
        latest_equals_saved=True,final_model_sha256=sha(saved)))
Path(__file__).with_name('completed_summary.json').write_text(json.dumps(results,indent=2))
(ROOT/'completed_summary.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))
