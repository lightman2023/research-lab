import os,sys,time,json,shutil,subprocess,hashlib
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
from flax import serialization
import numpy as np
work=Path(__file__).resolve().parent
source=Path('/home/user/chess/alphazero_chess_v2/experiments/search-budget-20261004/B200')
data=work/'frozen_data';data.mkdir(exist_ok=True)
files=sorted((source/'selfplay_data').glob('*.npz'))[-128:]
for path in files:shutil.copy2(path,data/path.name)
model=(source/'checkpoints/latest.msgpack').read_bytes();state=(source/'checkpoints/training_state.msgpack').read_bytes()
state_before=serialization.msgpack_restore(state);start_step=int(state_before['step'])
results={}
for label in ['baseline','candidate']:
    out=work/f'training_{label}';out.mkdir(exist_ok=True)
    (out/'latest.msgpack').write_bytes(model);(out/'state.msgpack').write_bytes(state)
    args=[sys.executable,'-u',str(work/label/'train.py'),'--steps','4','--batch-size','64',
        '--max-games','128','--data-dir',str(data),'--checkpoint',str(out/'latest.msgpack'),
        '--training-state',str(out/'state.msgpack'),'--learning-rate','0.0001','--constant-learning-rate','--seed','108009']
    begin=time.perf_counter()
    with (out/'run.log').open('w') as log:subprocess.run(args,env=dict(os.environ,JAX_PLATFORMS='cpu'),stdout=log,stderr=subprocess.STDOUT,check=True)
    seconds=time.perf_counter()-begin
    results[label]=dict(seconds_including_startup_compile=seconds,model_sha256=hashlib.sha256((out/'latest.msgpack').read_bytes()).hexdigest())
    print('TRAINED',label,seconds,flush=True)
old=serialization.msgpack_restore((work/'training_baseline/state.msgpack').read_bytes())
new=serialization.msgpack_restore((work/'training_candidate/state.msgpack').read_bytes())
max_diff=0.;leaves=0
def compare(a,b):
    global max_diff,leaves
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
        np.testing.assert_allclose(a,b,atol=1e-6,rtol=0)
        if a.size:max_diff=max(max_diff,float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))
        leaves+=1
compare(old,new);assert int(old['step'])==int(new['step'])==start_step+4
result=dict(device='cpu',source=str(source),source_model_sha256=hashlib.sha256(model).hexdigest(),source_state_sha256=hashlib.sha256(state).hexdigest(),
    starting_step=start_step,steps=4,batch_size=64,seed=108009,data_files=[p.name for p in files],runs=results,
    compared_leaves=leaves,max_absolute_difference=max_diff,optimizer_and_batch_stats_equal=True,
    speedup_including_compile=results['baseline']['seconds_including_startup_compile']/results['candidate']['seconds_including_startup_compile'])
(work/'training_validation.json').write_text(json.dumps(result,indent=2));print('VERIFIED training',max_diff,flush=True)
