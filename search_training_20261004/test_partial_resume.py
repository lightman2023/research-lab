import importlib.util
import json
import threading
from pathlib import Path

path=Path('C:/Users/User/Desktop/chess_v2/training_controller_search_B200.py')
spec=importlib.util.spec_from_file_location('resume',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Root:
    def after(self,delay,fn):fn()
class Var:
    def set(self,value):pass
def run(completed):
    c=m.Controller.__new__(m.Controller)
    c.root=Root();c.status_var=Var();c.cycles=6;c.games_total=48;c.stopping=threading.Event();c.calls=[]
    c.save_state=lambda:None;c.update_counter=lambda:None;c.append_log=lambda x:None;c.set_running=lambda x:None
    def stage(label,args):
        c.calls.append(args)
        if 'check_search_cycle.py' in args and 'before' in args:
            return json.dumps(dict(completed_game_numbers=completed,missing_game_numbers=[n for n in range(1,9) if n not in completed]))
        return ''
    c.run_stage=stage
    c.worker('continuous',(8,2,200,64,1,5))
    assert c.cycles==7 and c.games_total==56
    assert len([a for a in c.calls if 'train.py' in a])==1
    return [a for a in c.calls if 'parallel_selfplay.py' in a]
calls=run([1,2]);assert len(calls)==1
args=calls[0];assert args[args.index('--game-numbers')+1]=='3,4,5,6,7,8'
assert args[args.index('--seed')+1]=='20408004'
assert not run(list(range(1,9)))
print('PASS: saved games kept, missing six selected, full cycle skips regeneration, one training update per cycle.')
