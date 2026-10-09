from pathlib import Path
import shutil,hashlib
work=Path(__file__).resolve().parent;project=Path('/home/user/chess/alphazero_chess_v2')
name='monitor_experiment.py';target=project/name
assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256((work/'baseline'/name).read_bytes()).digest()
backup=target.with_name(name+'.pre-longterm-speedup-20261009')
if not backup.exists():shutil.copy2(target,backup)
for name in ['monitor_experiment.py','azchess/data_cache.py']:
    target=project/name;candidate=work/'candidate'/name
    compile(candidate.read_text(),str(target),'exec')
    temporary=target.with_name(target.name+'.speedup.tmp');shutil.copy2(candidate,temporary);temporary.replace(target)
print('Monitor input-file reuse and cached aggregate statistics deployed.')
