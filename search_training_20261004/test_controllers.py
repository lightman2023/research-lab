import importlib.util
import threading
import json
from pathlib import Path
import tkinter as tk

class Root:
    def after(self,delay,fn):fn()
class Var:
    def set(self,x):pass
records=[]
for branch,simulations in [('A50',50),('B200',200)]:
    path=Path(f'C:/Users/User/Desktop/chess_v2/training_controller_search_{branch}.py')
    spec=importlib.util.spec_from_file_location(branch,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tk.Tk();root.withdraw()
    gui=module.Controller(root)
    assert gui.cycles==0 and gui.simulations_var.get()==str(simulations)
    assert gui.settings()==(8,2,simulations,64,4,1)
    root.destroy()
    c=module.Controller.__new__(module.Controller)
    c.root=Root();c.status_var=Var();c.cycles=0;c.games_total=0;c.stopping=threading.Event();c.calls=[]
    c.save_state=lambda:None;c.update_counter=lambda:None;c.append_log=lambda x:None;c.set_running=lambda x:None
    c.run_stage=lambda label,args:c.calls.append(args)
    c.worker('continuous',(8,2,simulations,64,4,10))
    assert c.cycles==1 and c.games_total==8
    c.worker('continuous',(8,2,simulations,64,4,10))
    assert c.cycles==5 and c.games_total==40
    calls=len(c.calls);c.worker('continuous',(8,2,simulations,64,4,10));assert len(c.calls)==calls
    plays=[a for a in c.calls if 'parallel_selfplay.py' in a]
    trains=[a for a in c.calls if 'train.py' in a]
    monitors=[a for a in c.calls if 'monitor_search_training.py' in a]
    saves=[a for a in c.calls if a[0]=='cp']
    assert len(plays)==len(trains)==len(monitors)==5
    assert all(a[a.index('--simulations')+1]==str(simulations) for a in plays)
    assert all(a[a.index('--late-temperature')+1]=='1' for a in plays)
    assert all('--constant-learning-rate' in a and a[a.index('--learning-rate')+1]=='0.0001' for a in trains)
    assert len(saves)==2 and saves[-1][-1].endswith('cycle-000005.msgpack')
    c.generate(2,1,simulations,verify=True);assert c.games_total==40 and c.calls[-2][c.calls[-2].index('--output-dir')+1].endswith('validation_data')
    records.append(dict(branch=branch,initial_gate_one_cycle=True,total_stops_at_five=True,temperature_and_optimizer_forwarded=True,
        final_generation_saved=True,verify_data_excluded=True,seeds=[a[a.index('--seed')+1] for a in plays]))
assert records[0]['seeds']==records[1]['seeds']
Path(__file__).with_name('controller_validation.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records))
