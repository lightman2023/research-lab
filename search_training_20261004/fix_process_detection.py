from pathlib import Path
import shutil
win=Path('C:/Users/User/Desktop/chess_v2')
work=Path(__file__).resolve().parent
old="[p]arallel_selfplay.py|[/]selfplay.py|[/]train.py|[/]arena.py|[/]run_live.py|[/]monitor_experiment.py|[/]monitor_search_training.py"
new='[p]arallel_selfplay[.]py|[s]elfplay[.]py|[t]rain[.]py|[a]rena[.]py|[r]un_live[.]py|[m]onitor_experiment[.]py|[m]onitor_search_training[.]py|[c]heck_search_cycle[.]py'
for branch in ('A50','B200'):
    path=win/f'training_controller_search_{branch}.py'
    text=path.read_text(encoding='utf-8')
    assert text.count(old)==1
    shutil.copy2(path,path.with_name(path.name+'.pre-process-fix-20261005'))
    path.write_text(text.replace(old,new),encoding='utf-8')
    shutil.copy2(path,work/path.name)
path=win/'model_comparison.py'
text=path.read_text(encoding='utf-8')
old_compare="[p]arallel_selfplay.py|[/]selfplay.py|[/]train.py|[/]run_live.py|[/]monitor_experiment.py|[/]monitor_search_training.py"
assert text.count(old_compare)==1
shutil.copy2(path,path.with_name(path.name+'.pre-process-fix-20261005'))
path.write_text(text.replace(old_compare,new),encoding='utf-8')
shutil.copy2(win/'auto_search_training.py',win/'auto_search_training.py.pre-process-fix-20261005')
shutil.copy2(work/'auto_search_training.py',win/'auto_search_training.py')
print('Fixed process detection for relative and absolute script paths.')
