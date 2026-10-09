import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
project=Path('/home/user/chess/alphazero_chess_v2')
command=[sys.executable,str(project/'arena.py'),
    '--model-a',str(project/'experiments/live-lr0001-temp1-20261002/checkpoints/saved/cycle-000010.msgpack'),
    '--model-b',str(project/'experiments/search-budget-20261004/B200/checkpoints/saved/cycle-000005.msgpack'),
    '--games','1','--simulations-a','200','--simulations-b','200',
    '--max-plies','20','--device','cpu','--match-id','cpu-smoke-20261005']
start=perf_counter()
run=subprocess.run(command,cwd=project,capture_output=True,text=True)
Path(__file__).with_name('cpu_smoke.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stdout+run.stderr
events=[json.loads(line) for line in run.stdout.splitlines()]
assert events[0]['device']=='cpu' and all('cpu' in name.lower() for name in events[0]['actual_devices'])
summary=events[-1];assert summary['event']=='summary'
metadata=json.loads(Path(summary['metadata']).read_text())
assert metadata['device']=='cpu' and metadata['simulations_a']==metadata['simulations_b']==200
assert len(metadata['results'])==1
result=dict(cpu_only_verified=True,simulations_both=200,small_game_finished=True,
    seconds=perf_counter()-start,pgn=summary['pgn'],not_strength_evaluation=True)
Path(__file__).with_name('cpu_smoke_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
