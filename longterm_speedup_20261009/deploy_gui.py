import json,hashlib,shutil,importlib.util
from pathlib import Path
work=Path(__file__).resolve().parent;target=Path('C:/Users/User/Desktop/chess_v2/training_controller_search_B200.py')
source=work/'candidate/training_controller_search_B200.py'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(target)==digest(work/'baseline'/target.name)
spec=importlib.util.spec_from_file_location('gui',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Var:
    def __init__(self,value):self.value=value
    def get(self):return str(self.value)
c=module.Controller.__new__(module.Controller)
c.games_var=Var(8);c.workers_var=Var(8);c.simulations_var=Var(800);c.steps_var=Var(64);c.run_cycles_var=Var(0);c.save_interval_var=Var(5)
assert c.settings()==(8,8,800,64,0,5)
backup=target.with_name(target.name+'.pre-longterm-speedup-20261009')
if not backup.exists():shutil.copy2(target,backup)
shutil.copy2(source,target)
state_path=target.with_name('training_controller_search_B200_state.json')
state=json.loads(state_path.read_text(encoding='utf-8-sig'));state_backup=state_path.with_name(state_path.name+'.pre-longterm-speedup-20261009')
if not state_backup.exists():shutil.copy2(state_path,state_backup)
state['workers']='8'
temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(state_path)
(work/'gui_validation.json').write_text(json.dumps(dict(workers_8_accepted=True,default_saved_workers=8,cycles_preserved=state['cycles']),indent=2))
print('GUI updated, saved concurrent games=8; cycle and search settings retained.')
