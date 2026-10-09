import importlib.util,json,threading,tempfile
from pathlib import Path
from cycle_search_settings import cycle_search_settings
with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
    assert Path(directory).resolve().is_relative_to(Path(__file__).resolve().parent)
    first=cycle_search_settings(directory,45,800,6,'before')
    assert first['simulations_for_cycle']==200 and first['search_change_deferred']
    cycle_search_settings(directory,45,200,8,'after')
    fresh=cycle_search_settings(directory,46,800,0,'before')
    assert fresh['simulations_for_cycle']==800 and not fresh['search_change_deferred']
    resumed=cycle_search_settings(directory,46,200,2,'before')
    assert resumed['simulations_for_cycle']==800
    # Changing settings with zero saved games is safe.
    assert cycle_search_settings(directory,47,800,0,'before')['simulations_for_cycle']==800
    assert cycle_search_settings(directory,47,400,0,'before')['simulations_for_cycle']==400
spec=importlib.util.spec_from_file_location('gui','C:/Users/User/Desktop/chess_v2/training_controller_search_B200.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Var:
    def __init__(self,value=''):self.value=value
    def get(self):return str(self.value)
    def set(self,value):self.value=value
class Root:
    def after(self,delay,fn):fn()
c=m.Controller.__new__(m.Controller)
c.games_var=Var(8);c.workers_var=Var(2);c.simulations_var=Var(800);c.steps_var=Var(64);c.run_cycles_var=Var(2);c.save_interval_var=Var(5)
assert c.settings()==(8,2,800,64,2,5)
for value in [1,200,400,10000]:
    c.simulations_var.set(value);assert c.settings()[2]==value
for value in [0,10001]:
    c.simulations_var.set(value)
    try:c.settings()
    except ValueError:pass
    else:raise AssertionError('Invalid budget accepted')
c.cycles=44;c.games_total=352;c.root=Root();c.status_var=Var();c.stopping=threading.Event();c.calls=[];c.logs=[]
c.save_state=lambda:None;c.update_counter=lambda:None;c.append_log=c.logs.append;c.set_running=lambda x:None
def stage(label,args):
    c.calls.append(args)
    if 'check_search_cycle.py' in args and 'before' in args:
        cycle=int(args[args.index('--cycle')+1]);assert args[args.index('--simulations')+1]=='800'
        return json.dumps(dict(completed_game_numbers=list(range(1,7)) if cycle==45 else [],
            missing_game_numbers=[7,8] if cycle==45 else list(range(1,9)),simulations_for_cycle=200 if cycle==45 else 800))
    return ''
c.run_stage=stage;c.worker('continuous',(8,2,800,64,2,5))
assert c.cycles==46 and c.games_total==368
games=[a for a in c.calls if 'parallel_selfplay.py' in a]
assert [a[a.index('--simulations')+1] for a in games]==['200','800']
assert games[0][games[0].index('--game-numbers')+1]=='7,8'
assert len([a for a in c.calls if 'train.py' in a])==2
assert any('次のサイクルから800' in line for line in c.logs)
print('PASS: editable budgets; legacy partial cycle preserved; next cycle uses 800; interrupted 800 cycle keeps 800; invalid inputs rejected.')
