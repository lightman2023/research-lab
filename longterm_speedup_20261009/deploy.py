import os,json,hashlib,shutil
from pathlib import Path
work=Path(__file__).resolve().parent;project=Path('/home/user/chess/alphazero_chess_v2')
batch=json.loads((work/'batch_800_validation.json').read_text());history=json.loads((work/'history_validation.json').read_text())
training=json.loads((work/'training_validation.json').read_text());smoke=json.loads((work/'selfplay_validation.json').read_text())
assert history['input_exact'] and history['outcomes_exact'] and history['cases']>=3000
assert training['max_absolute_difference']==0 and smoke['all_original_npz_fields_exact'] and smoke['retry_does_not_duplicate']
assert batch['device']=='gpu' and batch['simulations']==800
for row in batch['results']:
    assert row['selected_move_equal'] and row['max_policy_total_variation']<=.005
    assert row['max_prior_difference']<.001 and row['speedup']>1
files=['azchess/mcts.py','azchess/checkpoint.py','azchess/history.py','azchess/batched.py','azchess/runtime.py','azchess/data_cache.py',
    'train.py','parallel_selfplay.py','batched_selfplay.py','check_search_cycle.py','verify_raw_visit_data.py']
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
for name in files:
    target=project/name;baseline=work/'baseline'/name
    if target.exists():assert baseline.exists() and digest(target)==digest(baseline),f'Current source changed: {name}'
    compile((work/'candidate'/name).read_text(),str(target),'exec')
records=[]
for name in files:
    target=project/name;backup=target.with_name(target.name+'.pre-longterm-speedup-20261009')
    before=digest(target) if target.exists() else None
    if target.exists() and not backup.exists():shutil.copy2(target,backup)
    temporary=target.with_name(target.name+'.speedup.tmp');shutil.copy2(work/'candidate'/name,temporary);os.replace(temporary,target)
    records.append(dict(file=str(target),backup=str(backup) if before else None,before=before,after=digest(target)))
(work/'deployment.json').write_text(json.dumps(dict(files=records,npz_or_model_files_modified=False),indent=2))
print('DEPLOYED',len(files),'source files. No model or existing NPZ changed.')
