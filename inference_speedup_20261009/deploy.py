from pathlib import Path
import shutil,hashlib,json,py_compile
work=Path(__file__).resolve().parent
project=Path('/home/user/chess/alphazero_chess_v2')
files=['azchess/batched.py','azchess/mcts.py']
for name in files:
    assert (project/name).read_bytes()==(work/'baseline'/name).read_bytes(),name
for name in files:
    target=project/name;backup=target.with_name(target.name+'.pre-inference-speedup-20261009')
    if not backup.exists():shutil.copy2(target,backup)
    shutil.copy2(work/'candidate'/name,target)
    py_compile.compile(str(target),doraise=True)
manifest={name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in files}
(work/'deployment.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
