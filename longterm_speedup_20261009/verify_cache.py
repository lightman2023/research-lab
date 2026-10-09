import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,time,json,tempfile,shutil,subprocess
from pathlib import Path
import numpy as np
work=Path(__file__).resolve().parent;sys.path.insert(0,str(work/'candidate'))
from azchess.data_cache import DataCache
with tempfile.TemporaryDirectory(dir=work) as directory:
    root=Path(directory);assert root.resolve().is_relative_to(work)
    files=sorted((work/'frozen_data').glob('*.npz'))[:4]
    for path in files:shutil.copy2(path,root/path.name)
    cache=DataCache(root);start=time.perf_counter()
    for path in root.glob('*.npz'):cache.get(path,validate=True)
    cold=time.perf_counter()-start;cache.flush()
    cache=DataCache(root);start=time.perf_counter()
    for path in root.glob('*.npz'):cache.get(path,validate=True)
    warm=time.perf_counter()-start;assert cache.hits==4 and cache.misses==0
    changed=root/files[0].name
    with np.load(changed) as data:arrays={key:data[key] for key in data.files}
    arrays['policy_values'][0]=-1
    np.savez_compressed(changed,**arrays)
    try:cache.get(changed,validate=True)
    except ValueError:pass
    else:raise AssertionError('Changed invalid data was accepted')
    # Full mode rechecks even an unchanged cached file.
    assert cache.get(root/files[1].name,validate=True,full=True)['validated']
    report=dict(files=4,cold_seconds=cold,cached_seconds=warm,changed_invalid_file_rejected=True,full_revalidation=True)
(work/'cache_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
