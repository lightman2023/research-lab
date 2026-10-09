import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import numpy as np
from flax import serialization
ROOT=Path('/home/user/chess/alphazero_chess_v2')
EXP=ROOT/'experiments/live-lr0001-temp1-20261002'
WORK=Path('/mnt/c/Users/User/Documents/ChatGPT/研究室/temperature1_gui_20261002')
WIN=Path('/mnt/c/Users/User/Desktop/chess_v2')
PYTHON=str(ROOT/'.venv/bin/python')
shutil.copy2(WORK/'monitor_experiment.py',ROOT/'monitor_experiment.py')
shutil.copy2(ROOT/'monitor_experiment.py',EXP/'code/monitor_experiment.py')
# Make the stop message describe the monitoring condition.
path=WIN/'training_controller_v2_temp1.py'
text=path.read_text(encoding='utf-8').replace('        if exit_code:\n            raise RuntimeError', '        if exit_code == 2 and last_line.startswith("MONITOR "):\n            raise RuntimeError("監視停止：価値ヘッドが不活性になる局面が10%以上に増えました。モデルと記録は保存済みです。")\n        if exit_code:\n            raise RuntimeError')
path.write_text(text,encoding='utf-8');shutil.copy2(path,WORK/path.name)
subprocess.run([PYTHON,str(WORK/'test_gui_config.py')],cwd=ROOT,check=True)
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [EXP/'checkpoints/latest.msgpack',EXP/'checkpoints/training_state.msgpack']}
smoke=EXP/'validation_smoke';smoke.mkdir()
subprocess.run([PYTHON,'-u','parallel_selfplay.py','--games','1','--workers','1','--simulations','2','--max-plies','40','--late-temperature','1','--seed','20281002','--experiment-cycle','0','--checkpoint',str(EXP/'checkpoints/latest.msgpack'),'--output-dir',str(smoke/'data')],cwd=ROOT,stdout=(smoke/'selfplay.log').open('w'),stderr=subprocess.STDOUT,check=True)
files=list((smoke/'data').glob('*.npz'));assert len(files)==1
with np.load(files[0]) as game:
    assert float(game['late_temperature'])==1 and int(game['seed'])==20281003 and int(game['experiment_cycle'])==0
    assert str(game['draw_claim_rule'])=='az_actual_threefold_50move_v1'
    smoke_plies=len(game['values'])
shutil.copy2(EXP/'checkpoints/latest.msgpack',smoke/'latest.msgpack')
shutil.copy2(EXP/'checkpoints/training_state.msgpack',smoke/'training_state.msgpack')
subprocess.run([PYTHON,'-u','train.py','--steps','1','--data-dir',str(smoke/'data'),'--checkpoint',str(smoke/'latest.msgpack'),'--training-state',str(smoke/'training_state.msgpack'),'--learning-rate','0.0001','--constant-learning-rate','--seed','20281002'],cwd=ROOT,stdout=(smoke/'train.log').open('w'),stderr=subprocess.STDOUT,check=True)
assert int(serialization.msgpack_restore((smoke/'training_state.msgpack').read_bytes())['step'])==2241
subprocess.run([PYTHON,'-u','monitor_experiment.py','--experiment',str(EXP),'--cycle','0'],cwd=ROOT,stdout=(EXP/'baseline_monitor.log').open('w'),stderr=subprocess.STDOUT,check=True)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
result=dict(selfplay_smoke_games=1,selfplay_smoke_plies=smoke_plies,temperature=1.,seed_forwarded=20281003,isolated_train_resumed_from=2240,isolated_train_saved_step=2241,main_model_and_state_unchanged=True,
            baseline_monitor=json.loads((EXP/'metrics/cycle-000000.json').read_text()),full_10_cycle_learning_started=False)
(EXP/'validation_results.json').write_text(json.dumps(result,indent=2));(WORK/'validation_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='baseline_monitor'}))
