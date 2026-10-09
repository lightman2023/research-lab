import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json
from pathlib import Path
from collections import Counter
import numpy as np
work=Path(__file__).resolve().parent;sys.path.insert(0,str(work/'candidate'))
from azchess.data_cache import DataCache
files=sorted((work/'frozen_data').glob('*.npz'));cache=DataCache(work/'frozen_data')
old_outcomes=Counter();new_outcomes=Counter();old_labels=Counter();new_labels=Counter()
for path in files:
    with np.load(path) as game:
        old_outcomes[str(game['termination'])]+=1
        for value in [-1,0,1]:old_labels[str(value)]+=int((game['values']==value).sum())
    info=cache.get(path);new_outcomes[info['termination']]+=1;new_labels.update(info['label_counts'])
assert old_outcomes==new_outcomes and old_labels==new_labels
cache.flush()
report=dict(files=len(files),outcomes_exact=True,labels_exact=True)
(work/'monitor_stats_validation.json').write_text(json.dumps(report));print(json.dumps(report))
