"""Independent inspection if the live experiment detects a dead value head."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys
from pathlib import Path
import json
sys.path.insert(0,'/home/user/chess/alphazero_chess_v2')
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from azchess.checkpoint import load_variables
from train import ReplayBuffer
ROOT=Path('/home/user/chess/alphazero_chess_v2/experiments/live-lr0001-20261002-run2')
model,variables=load_variables(ROOT/'checkpoints/saved/cycle-000002.msgpack')
replay=ReplayBuffer(sorted((ROOT/'selfplay_data').glob('*.npz'))[:16],64,seed=20261002)
x,p,z=map(jnp.asarray,replay.sample(128))
results={}
for train in [False,True]:
    (_,values),capture=model.apply(variables,x,train=train,capture_intermediates=lambda mod,method:isinstance(mod,nn.Dense),mutable=['intermediates','batch_stats'])
    dense=np.asarray(capture['intermediates']['Dense_0']['__call__'][0]); values=np.asarray(values)
    results[str(train)]=dict(dead=int(np.all(dense<=0,axis=1).sum()),positions=len(values),max_preactivation_quantiles=np.quantile(dense.max(axis=1),[0,.25,.5,.75,1]).tolist(),value_std=float(values.std()),value_mse=float(np.mean((values-np.asarray(z))**2)))
def loss(params):
    (logits,value),_=model.apply(dict(params=params,batch_stats=variables['batch_stats']),x,train=True,mutable=['batch_stats'])
    return jnp.mean((value-z)**2)
grads=jax.grad(loss)(variables['params'])
results['value_loss_gradient_norms']={name:{field:float(jnp.linalg.norm(value)) for field,value in grads[name].items()} for name in ['Dense_0','Dense_1']}
results['labels']=dict(zip(*[list(map(str,np.unique(np.asarray(z),return_counts=True)[0])),list(map(int,np.unique(np.asarray(z),return_counts=True)[1]))]))
(ROOT/'collapse_inspection.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
