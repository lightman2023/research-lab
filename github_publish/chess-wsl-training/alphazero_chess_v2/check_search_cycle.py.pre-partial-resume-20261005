import argparse
import json
from pathlib import Path
import numpy as np
from flax import serialization

parser=argparse.ArgumentParser()
parser.add_argument('--experiment',type=Path,required=True)
parser.add_argument('--cycle',type=int,required=True)
parser.add_argument('--stage',choices=['before','after'],required=True)
args=parser.parse_args()
root=args.experiment
step=int(serialization.msgpack_restore((root/'checkpoints/training_state.msgpack').read_bytes())['step'])
expected_step=2880+(args.cycle-1)*64
if step!=expected_step:
    raise SystemExit(f'Cycle/state mismatch: expected step {expected_step}, found {step}. Preserve data and inspect interrupted run before restarting.')
files=list((root/'selfplay_data').glob('*.npz'))
expected_games=160+8*(args.cycle-1 if args.stage=='before' else args.cycle)
if len(files)!=expected_games:
    raise SystemExit(f'Data count mismatch: expected {expected_games}, found {len(files)}. Preserve existing data; do not regenerate interrupted cycle.')
new=[]
for path in files:
    if not path.name.startswith('game-'):continue
    with np.load(path) as game:
        if int(game['experiment_cycle'])==args.cycle:new.append(path.name)
expected_new=0 if args.stage=='before' else 8
if len(new)!=expected_new:
    raise SystemExit(f'New cycle games mismatch: expected {expected_new}, found {len(new)}. Inspect interrupted cycle before restarting.')
print(json.dumps(dict(stage=args.stage,cycle=args.cycle,training_step=step,data_games=len(files),new_cycle_games=len(new))),flush=True)
