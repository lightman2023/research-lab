import json
from pathlib import Path
from collections import Counter
root=Path('/home/user/chess/alphazero_chess_v2/experiments/live-lr0001-20261002-run2')
p=json.loads((root/'progress.json').read_text())
print(json.dumps(dict(cycles=len(p['history']),games=len(p['games']),terminations=dict(Counter(r['termination'] for r in p['games'])),final_dead=p['history'][-1]['evaluation']['dead'],value_mse=p['history'][-1]['evaluation']['value_mse'],probe_counts={str(t):len(list((root/f'temperature-{t}').glob('*.npz'))) for t in [0.25,1.0]})))
