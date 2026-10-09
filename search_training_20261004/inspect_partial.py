import json
from pathlib import Path
from collections import Counter
import numpy as np
from flax import serialization
root=Path('/home/user/chess/alphazero_chess_v2/experiments/search-budget-20261004/B200')
step=int(serialization.msgpack_restore((root/'checkpoints/training_state.msgpack').read_bytes())['step'])
games=[]
for path in (root/'selfplay_data').glob('game-*.npz'):
    with np.load(path) as item:
        if int(item['experiment_cycle'])==7:
            games.append(dict(file=path.name,cycle=int(item['experiment_cycle']),seed=int(item['seed']),keys=item.files,
                termination=str(item['termination']),plies=len(item['values'])))
print(json.dumps(dict(training_step=step,partial_games=games)))
