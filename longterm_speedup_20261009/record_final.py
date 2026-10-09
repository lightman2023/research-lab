import json,hashlib
from pathlib import Path
work=Path(__file__).resolve().parent;project=Path('/home/user/chess/alphazero_chess_v2')
files=['azchess/mcts.py','azchess/checkpoint.py','azchess/history.py','azchess/batched.py','azchess/runtime.py','azchess/data_cache.py',
    'train.py','parallel_selfplay.py','batched_selfplay.py','check_search_cycle.py','verify_raw_visit_data.py','monitor_experiment.py']
records=[]
for name in files:
    path=project/name;expected=work/'candidate'/name
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==hashlib.sha256(expected.read_bytes()).hexdigest()
    records.append(dict(file=str(path),sha256=digest))
note=(work/'readme_append.md').read_text();readme=project/'README.md';text=readme.read_text()
if '長期学習向け追加高速化' not in text:readme.write_text(text+note)
(work/'final_deployment.json').write_text(json.dumps(dict(files=records,original_npz_and_model_edits=False),indent=2))
print('Final source hashes verified; project README updated.')
