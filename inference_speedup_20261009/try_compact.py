import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import sys,importlib,importlib.util,time,json
from pathlib import Path
import numpy as np,chess,jax,jax.numpy as jnp
work=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('candidate',work/'candidate/azchess/__init__.py',submodule_search_locations=[str(work/'candidate/azchess')])
module=importlib.util.module_from_spec(spec);sys.modules['candidate']=module;spec.loader.exec_module(module)
game=importlib.import_module('candidate.game')
model,variables=importlib.import_module('candidate.checkpoint').load_variables(Path('/home/user/chess/alphazero_chess_v2/experiments/search-budget-20261004/B200/checkpoints/saved/cycle-000050.msgpack'))
variables=jax.device_put(variables)
board=chess.Board();states=np.stack([game.encode_board(board)]*8)
indices=np.zeros((8,218),dtype=np.int32)
mapping=game.legal_action_map(board);used=len(mapping)
indices[:,:used]=np.array(list(mapping))
@jax.jit
def full(variables,states):
    with jax.default_matmul_precision('highest'):
        return model.apply(variables,states,train=False)
@jax.jit
def compact(variables,states,indices):
    with jax.default_matmul_precision('highest'):
        logits,values=model.apply(variables,states,train=False)
    return jnp.take_along_axis(logits,indices,axis=1),values
expected=jax.device_get(full(variables,states));actual=jax.device_get(compact(variables,states,indices))
np.testing.assert_array_equal(expected[0][:,list(mapping)],actual[0][:,:used]);np.testing.assert_array_equal(expected[1],actual[1])
rows=[]
for repeat in range(4):
    times={}
    for label in (('full','direct','compact') if repeat%2==0 else ('compact','direct','full')):
        start=time.perf_counter()
        for i in range(200):
            if label=='compact':jax.device_get(compact(variables,states,indices))
            elif label=='full':jax.device_get(full(variables,jnp.asarray(states)))
            else:jax.device_get(full(variables,states))
        times[label]=time.perf_counter()-start
    rows.append(times);print(json.dumps(times),flush=True)
(work/'dispatch_validation.json').write_text(json.dumps(dict(repeats=rows,iterations=200,workers=8,outputs_exact=True),indent=2))
