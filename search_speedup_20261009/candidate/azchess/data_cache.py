"""Disposable metadata/validation cache; original NPZ files stay authoritative."""
import json,os
from pathlib import Path
import numpy as np

class DataCache:
    def __init__(self,directory):
        self.path=Path(directory)/'.npz_metadata_v2.json'
        try:self.data=json.loads(self.path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):self.data={}
        if self.data.get('version')!=2:self.data=dict(version=2,entries={},validation_runs=0)
        self.hits=0;self.misses=0
    def get(self,path,validate=False,full=False):
        path=Path(path);stat=path.stat();signature=[stat.st_size,stat.st_mtime_ns,stat.st_ino]
        record=self.data['entries'].get(path.name)
        if record and record['signature']==signature and 'label_counts' in record and (not validate or record.get('validated')) and not full:
            self.hits+=1;return record
        with np.load(path) as game:
            known_validated=bool(record and record['signature']==signature and record.get('validated'))
            labels=game['values']
            record=dict(signature=signature,positions=len(labels),label_counts={str(v):int((labels==v).sum()) for v in [-1,0,1]},
                cycle=int(game['experiment_cycle']) if 'experiment_cycle' in game else -1,
                seed=int(game['seed']) if 'seed' in game else -1,
                termination=str(game['termination']) if 'termination' in game else 'unknown',validated=known_validated)
            if validate:
                if 'policy_target_kind' not in game or str(game['policy_target_kind'])!='raw_mcts_visits_v1':
                    raise ValueError(f'Wrong policy target kind: {path}')
                moves=game['moves'];states=game['states'];offsets=game['policy_offsets'];values=game['policy_values'].astype(np.float32)
                if len(moves)!=len(states) or len(offsets)!=len(moves)+1 or len(moves)!=record['positions']:
                    raise ValueError(f'Position count mismatch: {path}')
                if offsets[0]!=0 or offsets[-1]!=len(values) or np.any(np.diff(offsets)<=0):
                    raise ValueError(f'Invalid policy offsets: {path}')
                if not np.isfinite(values).all() or (values<0).any() or not np.isfinite(states).all():
                    raise ValueError(f'Invalid state/policy values: {path}')
                if not np.allclose(np.add.reduceat(values,offsets[:-1]),1.,atol=.01):
                    raise ValueError(f'Policy targets do not sum to one: {path}')
                record['validated']=True
        self.data['entries'][path.name]=record;self.misses+=1;return record
    def flush(self,files=None):
        if files is not None:
            names={Path(p).name for p in files}
            self.data['entries']={k:v for k,v in self.data['entries'].items() if k in names}
        temporary=self.path.with_name(self.path.name+f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(self.data,separators=(',',':')))
        temporary.replace(self.path)
