import hashlib
import json
from pathlib import Path
from collections import Counter
from flax import serialization
root=Path('/home/user/chess/alphazero_chess_v2/experiments/live-lr0001-temp1-20261002')
rows=[json.loads((root/f'metrics/cycle-{i:06d}.json').read_text()) for i in range(11)]
outcomes=Counter(); games=[]
for row in rows[1:]:
    assert len(row['cycle_games'])==8
    outcomes.update(row['cycle_outcomes']); games.extend(row['cycle_games'])
assert len(games)==len(set(games))==80
step=int(serialization.msgpack_restore((root/'checkpoints/training_state.msgpack').read_bytes())['step'])
assert step==2880
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(root/'checkpoints/latest.msgpack')==digest(root/'checkpoints/saved/cycle-000010.msgpack')
result=dict(completed_cycles=10,new_selfplay_games=80,added_updates=640,final_training_step=step,new_game_outcomes=dict(outcomes),
            fixed_dead_counts=[r['fixed']['dead'] for r in rows],live_dead_counts=[r['live_training_sample']['dead'] for r in rows],
            fixed_value_mse_start=rows[0]['fixed']['value_mse'],fixed_value_mse_end=rows[-1]['fixed']['value_mse'],
            latest_matches_final_saved_model=True,final_model_sha256=digest(root/'checkpoints/latest.msgpack'))
(root/'completed_summary.json').write_text(json.dumps(result,indent=2))
work=Path('/mnt/c/Users/User/Documents/ChatGPT/研究室/temperature1_gui_20261002')
(work/'completed_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
