import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tkinter as tk

path=Path('C:/Users/User/Desktop/chess_v2/training_controller_search_B200.py')
spec=importlib.util.spec_from_file_location('continuation',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=tk.Tk();root.withdraw();app=m.Controller(root)
assert app.cycles==5 and app.run_cycles_var.get()=='0'
assert '[a]rena' not in m.Controller.begin.__code__.co_consts.__str__()
root.destroy()
class Root:
    def after(self,delay,fn):fn()
class Var:
    def set(self,value):pass
def controller():
    c=m.Controller.__new__(m.Controller);c.root=Root();c.status_var=Var();c.cycles=5;c.games_total=40
    c.stopping=threading.Event();c.calls=[]
    c.save_state=lambda:None;c.update_counter=lambda:None;c.append_log=lambda x:None;c.set_running=lambda x:None
    c.run_stage=lambda label,args:c.calls.append(args)
    return c
c=controller();c.worker('continuous',(8,2,200,64,3,10))
assert c.cycles==8 and c.games_total==64
trains=[a for a in c.calls if 'train.py' in a]
saves=[a for a in c.calls if a[0]=='cp']
assert len(trains)==3 and len(saves)==1 and saves[0][-1].endswith('cycle-000008.msgpack')
assert all('--constant-learning-rate' in a for a in trains)
c=controller()
def stage(label,args):
    c.calls.append(args)
    if 'monitor_search_training.py' in args and c.cycles==7:c.stopping.set()
c.run_stage=stage;c.worker('continuous',(8,2,200,64,0,1))
assert c.cycles==7 and len([a for a in c.calls if 'train.py' in a])==2
sys.path.insert(0,'C:/Users/User/Desktop/chess_v2')
import model_comparison as comparison
test=comparison.ModelComparisonApp.__new__(comparison.ModelComparisonApp)
allowed=f'{comparison.PROJECT}/experiments/search-budget-20261004/B200'
test.run_wsl=lambda *a,**k:SimpleNamespace(returncode=0,stdout=f'10 python -u train.py --checkpoint {allowed}/checkpoints/latest.msgpack')
assert not test.training_running()
test.run_wsl=lambda *a,**k:SimpleNamespace(returncode=0,stdout='10 python -u train.py --checkpoint /other/latest.msgpack')
assert test.training_running()
result=dict(resume_beyond_five=True,zero_runs_until_stopped=True,finite_final_snapshot_saved=True,
    evaluation_allows_B_training=True,unrelated_training_blocked=True,training_started=False)
Path(__file__).with_name('continuation_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
