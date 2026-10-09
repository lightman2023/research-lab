import os,json,hashlib,shutil
from pathlib import Path
work=Path(__file__).resolve().parent
project=Path('/home/user/chess/alphazero_chess_v2')
gpu=json.loads((work/'search_gpu_validation.json').read_text())
training=json.loads((work/'training_validation.json').read_text())
rules=json.loads((work/'rules_validation.json').read_text())
assert gpu['device']=='gpu' and len(gpu['checks'])==4
assert training['max_absolute_difference']==0 and rules['identical']
files=['azchess/mcts.py','azchess/game.py','train.py']
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
for name in files:
    assert digest(project/name)==digest(work/'baseline'/name),f'Current code changed; inspect before updating: {name}'
records=[]
for name in files:
    target=project/name;backup=target.with_name(target.name+'.pre-speedup-20261009')
    if not backup.exists():shutil.copy2(target,backup)
    candidate=work/'candidate'/name
    compile(candidate.read_text(),str(candidate),'exec')
    temporary=target.with_name(target.name+'.speedup.tmp')
    shutil.copy2(candidate,temporary);os.replace(temporary,target)
    records.append(dict(file=str(target),backup=str(backup),before=digest(backup),after=digest(target)))
(work/'deployment.json').write_text(json.dumps(dict(files=records,model_or_training_data_modified=False,
    application='New self-play/train processes load the changes; already running games finish with loaded code.'),indent=2))
print('DEPLOYED',len(records),'source files; checkpoint/data unchanged.')
