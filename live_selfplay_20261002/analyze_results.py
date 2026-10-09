import json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path('/home/user/chess/alphazero_chess_v2/experiments/live-lr0001-20261002-run2')
data=json.loads((ROOT/'summary.json').read_text())
def aggregate(rows):
    return dict(games=len(rows),terminations=dict(Counter(r['termination'] for r in rows)),
                mean_plies=float(np.mean([r['plies'] for r in rows])),
                median_plies=float(np.median([r['plies'] for r in rows])),
                total_reversals=sum(r['reversals'] for r in rows),
                pooled_reversal_rate=sum(r['reversals'] for r in rows)/sum(max(1,r['plies']-2) for r in rows))
labels=Counter()
for path in (ROOT/'selfplay_data').glob('*.npz'):
    with np.load(path) as item:
        labels.update({str(v):int((item['values']==v).sum()) for v in [-1,0,1]})
stats=dict(training=aggregate(data['games']),labels=dict(labels),probes={name:aggregate(rows) for name,rows in data['probes'].items()},baseline=data['baseline'],final=data['history'][-1],validated=data['validated_npz_and_pgn'])
if data['probes']:
    a=data['probes']['temperature-0.25']; b=data['probes']['temperature-1.0']
    stats['paired_seeds_match']=all(x['seed']==y['seed'] for x,y in zip(a,b))
    stats['paired_plies_differences']=[y['plies']-x['plies'] for x,y in zip(a,b)]
(ROOT/'analysis.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
