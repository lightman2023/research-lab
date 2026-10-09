import shutil,hashlib
from pathlib import Path
work=Path(__file__).resolve().parent;target=Path('C:/Users/User/Desktop/chess_v2/training_controller_search_B200.py')
assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256((work/'gui_before_timings.py').read_bytes()).digest()
source=work/'candidate'/target.name;compile(source.read_text(encoding='utf-8'),str(source),'exec');shutil.copy2(source,target)
print('GUI logs per-stage and complete-cycle elapsed times.')
