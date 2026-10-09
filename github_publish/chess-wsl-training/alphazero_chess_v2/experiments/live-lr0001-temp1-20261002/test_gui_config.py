import importlib.util
import subprocess
import threading
import json
from pathlib import Path
subprocess.CREATE_NO_WINDOW=0
spec=importlib.util.spec_from_file_location('gui','/mnt/c/Users/User/Desktop/chess_v2/training_controller_v2_temp1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Root:
    def after(self,delay,callback): callback()
class Var:
    def set(self,value): pass
def controller():
    c=m.Controller.__new__(m.Controller); c.root=Root(); c.status_var=Var()
    c.cycles=0;c.games_total=0;c.stopping=threading.Event();c.calls=[]
    c.save_state=lambda:None;c.update_counter=lambda:None;c.append_log=lambda text:None;c.set_running=lambda value:None
    c.run_stage=lambda label,args:c.calls.append(args)
    return c
c=controller();c.worker('continuous',(8,2,50,64,10,1))
trains=[a for a in c.calls if 'train.py' in a]
plays=[a for a in c.calls if 'parallel_selfplay.py' in a]
monitors=[a for a in c.calls if 'monitor_experiment.py' in a]
assert c.cycles==10 and c.games_total==80 and len(trains)==len(plays)==len(monitors)==10
assert all(a[a.index('--learning-rate')+1]=='0.0001' and '--constant-learning-rate' in a for a in trains)
assert all(a[a.index('--late-temperature')+1]=='1' for a in plays)
assert len({a[a.index('--seed')+1] for a in plays})==10
v=controller();v.generate(2,1,50,verify=True)
assert v.games_total==0 and v.calls[0][v.calls[0].index('--output-dir')+1].endswith('/validation_data')
# A longer save interval still saves the final finite run model.
c=controller();c.worker('continuous',(8,2,50,64,10,25))
saves=[a for a in c.calls if a[0]=='cp'];assert len(saves)==1 and saves[0][-1].endswith('cycle-000010.msgpack')
result=dict(mocked_10_cycles=True,temperature_forwarded=True,constant_optimizer_args=True,monitor_each_cycle=True,verify_data_separate=True,final_saved_when_interval_exceeds_run=True)
Path(__file__).with_name('gui_config_test_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
