import sys
import json
import subprocess
from pathlib import Path
root=Path('/home/user/chess/alphazero_chess_v2')
for branch in ('A50','B200'):
    command=[sys.executable,str(root/'check_search_cycle.py'),'--experiment',str(root/'experiments/search-budget-20261004'/branch),'--cycle','1']
    before=subprocess.run(command+['--stage','before'],capture_output=True,text=True)
    assert before.returncode==0,before.stdout+before.stderr
    after=subprocess.run(command+['--stage','after'],capture_output=True,text=True)
    assert after.returncode!=0 and 'expected 168, found 160' in after.stderr,after.stdout+after.stderr
result=dict(initial_cycle_preflight_passes=True,incomplete_new_cycle_rejected=True,branches_unchanged=True)
Path(__file__).with_name('cycle_guard_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
