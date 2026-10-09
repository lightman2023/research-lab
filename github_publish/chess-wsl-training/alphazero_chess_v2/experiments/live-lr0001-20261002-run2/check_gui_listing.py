import importlib.util
import subprocess
import json
from pathlib import Path
subprocess.CREATE_NO_WINDOW=0
spec=importlib.util.spec_from_file_location('comparison','/mnt/c/Users/User/Desktop/chess_v2/model_comparison.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
app=module.ModelComparisonApp.__new__(module.ModelComparisonApp)
class Root:
    def after(self,delay,callback): callback()
app.root=Root()
app.run_wsl=lambda command,**kwargs:subprocess.run(['bash','-lc',command],text=True,**kwargs)
output={}
def record(names,paths,code):
    assert code==0
    low=[name for name in names if name.startswith('低学習率・自己対局 / ')]
    assert '低学習率・自己対局 / cycle-000000.msgpack' in low
    assert '低学習率・自己対局 / cycle-000002.msgpack' in low
    output['new_experiment_models']=low
app.show_models=record
app.refresh_models_worker()
output['busy_during_live_training']=app.training_running()
assert output['busy_during_live_training']
Path(__file__).with_name('gui_listing_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
print(json.dumps(output,ensure_ascii=False))
