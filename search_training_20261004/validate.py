import sys
import subprocess
import shutil
import json
import hashlib
from pathlib import Path

ROOT=Path('/home/user/chess/alphazero_chess_v2');sys.path.insert(0,str(ROOT))
from flax import serialization
import numpy as np
DEST=ROOT/'experiments/search-budget-20261004'
config=json.loads((DEST/'configuration.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((DEST/'carried_data_manifest.json').read_text())
for branch in ('A50','B200'):
    dest=DEST/branch
    assert digest(dest/'checkpoints/latest.msgpack')==config['checkpoint_sha256']
    assert digest(dest/'checkpoints/training_state.msgpack')==config['state_sha256']
    assert len(list((dest/'selfplay_data').glob('*.npz')))==160
    for entry in manifest:
        assert digest(dest/'selfplay_data'/entry['name'])==entry['sha256']
    if not (dest/'metrics/cycle-000000.json').exists():
        run=subprocess.run([sys.executable,str(ROOT/'monitor_search_training.py'),'--experiment',str(dest),'--cycle','0'],cwd=ROOT,capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr
        (DEST/f'baseline-{branch}.log').write_text(run.stdout+run.stderr)
a=json.loads((DEST/'A50/metrics/cycle-000000.json').read_text())
b=json.loads((DEST/'B200/metrics/cycle-000000.json').read_text())
differences=[]
def compare(left,right,key=''):
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        for name in left:compare(left[name],right[name],key+'/'+name)
    elif isinstance(left,list):
        assert len(left)==len(right)
        for index,(x,y) in enumerate(zip(left,right)):compare(x,y,key+'/'+str(index))
    elif isinstance(left,float):
        assert np.isclose(left,right,atol=1e-3,rtol=1e-3),(key,left,right)
        differences.append((abs(left-right),key))
    else:assert left==right,(key,left,right)
compare(a,b)
assert a['cycle_games']==[] and a['cycle_outcomes']=={}
assert a['training_step']==2880 and len(a['repetition_policy_probe'])==6
smoke=DEST/'validation_smoke';smoke.mkdir()
shutil.copy2(DEST/'source/latest.msgpack',smoke/'latest.msgpack')
shutil.copy2(DEST/'source/training_state.msgpack',smoke/'training_state.msgpack')
# Real one-update check on a disposable copy. The main A/B branches remain at cycle zero.
command=[sys.executable,str(ROOT/'train.py'),'--steps','1','--batch-size','64',
    '--data-dir',str(DEST/'A50/selfplay_data'),'--checkpoint',str(smoke/'latest.msgpack'),
    '--training-state',str(smoke/'training_state.msgpack'),'--learning-rate','0.0001',
    '--constant-learning-rate','--seed','20401004']
run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
(smoke/'train.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stdout+run.stderr
assert int(serialization.msgpack_restore((smoke/'training_state.msgpack').read_bytes())['step'])==2881
for branch in ('A50','B200'):
    assert digest(DEST/branch/'checkpoints/latest.msgpack')==config['checkpoint_sha256']
    assert digest(DEST/branch/'checkpoints/training_state.msgpack')==config['state_sha256']
result=dict(copied_games_each=160,data_hashes_identical=True,model_and_optimizer_identical=True,
    starting_step=2880,baseline_metrics_agree_with_tolerance=True,
    baseline_max_abs_difference=max(differences)[0],baseline_float_atol=1e-3,baseline_float_rtol=1e-3,
    fixed_dead=a['fixed']['dead'],
    repetition_positions=6,carried_games_not_counted_as_new=True,
    real_update_on_copy=2881,main_branches_unchanged=True,training_cycles_completed=0)
Path(__file__).with_name('validation_results.json').write_text(json.dumps(result,indent=2))
(DEST/'validation_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
