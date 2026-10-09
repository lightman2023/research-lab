import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from flax import serialization
project=Path('/home/user/chess/alphazero_chess_v2')
rows=[]
for branch in ('A50','B200'):
    root=project/'experiments/search-budget-20261004'/branch
    counts=Counter();seeds=set();carried=0
    for path in sorted((root/'selfplay_data').glob('*.npz')):
        if path.name.startswith('carried-'):carried+=1;continue
        with np.load(path) as data:
            cycle=int(data['experiment_cycle']);seed=int(data['seed'])
            assert (cycle,seed) not in seeds,('duplicate seed',branch,cycle,seed)
            seeds.add((cycle,seed));counts[cycle]+=1
    assert carried==160 and all(n<=8 for n in counts.values()),(branch,counts)
    step=int(serialization.msgpack_restore((root/'checkpoints/training_state.msgpack').read_bytes())['step'])
    rows.append(dict(branch=branch,carried=carried,new_games_by_cycle=dict(counts),step=step,duplicate_cycle_seed=False))
Path(__file__).with_name('error_recovery_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
