from pathlib import Path
import shutil,json
work=Path(__file__).resolve().parent
target=Path('C:/Users/User/Desktop/chess_v2/training_controller_search_B200.py')
text=target.read_text(encoding='utf-8')
assert 'self.workers_var = tk.StringVar(value=str(state.get("workers", 2)))' in text
shutil.copy2(work/'candidate'/target.name,target)
state_path=target.with_name('training_controller_search_B200_state.json')
state=json.loads(state_path.read_text(encoding='utf-8-sig'));state['schema_version']=3;state['workers']='8'
temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(state_path)
print('GUI migration retains the 8-game default when an older running GUI saves its version-2 state.')
