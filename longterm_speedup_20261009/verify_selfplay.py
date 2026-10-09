import os,sys,json,subprocess,hashlib
from pathlib import Path
import numpy as np
work=Path(__file__).resolve().parent
env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
checkpoint='/home/user/chess/alphazero_chess_v2/experiments/search-budget-20261004/B200/checkpoints/saved/cycle-000050.msgpack'
source=work/'baseline';candidate=work/'candidate'
old_dir=work/'smoke_legacy';new_dir=work/'smoke_batched'
old_dir.mkdir(exist_ok=True);new_dir.mkdir(exist_ok=True)
def run(args,log):
    with (work/log).open('w') as f:subprocess.run([sys.executable,'-u']+args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
common=['--simulations','64','--max-plies','6','--checkpoint',checkpoint,'--late-temperature','1','--experiment-cycle','99109']
for number in [1,3]:
    run([str(source/'selfplay.py'),'--games','1','--output-dir',str(old_dir),'--seed',str(90109+number)]+common,f'smoke_legacy_{number}.log')
args=[str(candidate/'parallel_selfplay.py'),'--games','3','--game-numbers','1,3','--workers','2',
    '--engine','batched','--output-dir',str(new_dir),'--seed','90109']+common
run(args,'smoke_batched.log')
def games(directory):
    result={}
    for path in directory.glob('*.npz'):
        with np.load(path) as data:result[int(data['seed'])]={k:data[k] for k in data.files}
    return result
before=games(old_dir);after=games(new_dir)
assert before.keys()==after.keys()=={90110,90112}
for seed in before:
    for key,array in before[seed].items():np.testing.assert_array_equal(array,after[seed][key])
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in new_dir.glob('*.npz')}
run(args,'smoke_batched_resume.log')
assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in new_dir.glob('*.npz')}
run([str(candidate/'verify_raw_visit_data.py'),'--data-dir',str(new_dir),'--min-games','2'],'smoke_validate.log')
report=dict(device='cpu',games=2,seeds=[90110,90112],plies_per_game=6,simulations=64,
    all_original_npz_fields_exact=True,partial_game_numbers_preserved=True,retry_does_not_duplicate=True)
(work/'selfplay_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
