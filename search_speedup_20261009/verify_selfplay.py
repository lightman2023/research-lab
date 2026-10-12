import os,sys,json,subprocess,shutil
from pathlib import Path
import numpy as np
work=Path(__file__).resolve().parent
project=Path('/home/user/chess/alphazero_chess_v2')
env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
for label in ('baseline','candidate'):
    shutil.copy2(project/'batched_selfplay.py',work/label/'batched_selfplay.py')
    output=work/('smoke_final_'+label);output.mkdir(exist_ok=True)
    args=[sys.executable,str(work/label/'batched_selfplay.py'),'--games','2','--workers','2',
          '--simulations','64','--max-plies','12','--checkpoint',str(project/'experiments/search-budget-20261004/B200/checkpoints/saved/cycle-000050.msgpack'),
          '--output-dir',str(output),'--seed','90109','--experiment-cycle','991010']
    with (work/('smoke_'+label+'.log')).open('w') as log:
        subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def load(label):
    games={}
    for path in (work/('smoke_final_'+label)).glob('*.npz'):
        with np.load(path) as data:games[int(data['seed'])]={k:data[k] for k in data.files}
    return games
before,after=load('baseline'),load('candidate')
assert before.keys()==after.keys()=={90110,90111}
for seed in before:
    assert before[seed].keys()==after[seed].keys()
    for key in before[seed]:np.testing.assert_array_equal(before[seed][key],after[seed][key])
report=dict(games=2,plies=12,simulations=64,workers=2,device='cpu',all_npz_fields_exact=True)
(work/'selfplay_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
