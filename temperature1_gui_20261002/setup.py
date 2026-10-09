import hashlib
import json
import shutil
from pathlib import Path
ROOT=Path('/home/user/chess/alphazero_chess_v2')
WORK=Path('/mnt/c/Users/User/Documents/ChatGPT/研究室/temperature1_gui_20261002')
SOURCE=ROOT/'experiments/live-lr0001-20261002-run2'
DEST=ROOT/'experiments/live-lr0001-temp1-20261002'
WIN=Path('/mnt/c/Users/User/Desktop/chess_v2')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
if DEST.exists(): raise RuntimeError(f'Experiment already exists: {DEST}')
DEST.mkdir(); (DEST/'checkpoints/saved').mkdir(parents=True); (DEST/'selfplay_data').mkdir()
for source,target in [('checkpoints/latest.msgpack','checkpoints/latest.msgpack'),('checkpoints/latest.msgpack','checkpoints/saved/cycle-000000.msgpack'),('checkpoints/training_state.msgpack','checkpoints/training_state.msgpack'),('evaluation_manifest.json','evaluation_manifest.json')]:
    shutil.copy2(SOURCE/source,DEST/target)
manifest=[]
for path in sorted((SOURCE/'selfplay_data').glob('*.npz')):
    copied=DEST/'selfplay_data'/path.name; shutil.copy2(path,copied)
    manifest.append(dict(source=str(path),copy=str(copied),sha256=digest(copied)))
assert len(manifest)==80
config=dict(source_experiment=str(SOURCE),source_checkpoint_sha256=digest(SOURCE/'checkpoints/latest.msgpack'),source_state_sha256=digest(SOURCE/'checkpoints/training_state.msgpack'),starting_training_step=2240,
            carried_games=80,learning_rate=.0001,constant_learning_rate=True,late_temperature=1.,simulations=50,max_plies=512,games_per_cycle=8,updates_per_cycle=64,batch_size=64,cycles_per_run=10,save_interval=1,
            seed_base=20281002,draw_rule='az_actual_threefold_50move_v1',policy_target_kind='raw_mcts_visits_v1',monitor_stop_fraction=.1,
            note='Only carried training games copied; preceding 16 sampling probe games are excluded. Baseline cycle0 is preceding cycle10. Historical optimizer state is preserved.')
(DEST/'configuration.json').write_text(json.dumps(config,indent=2))
(DEST/'carried_data_manifest.json').write_text(json.dumps(manifest,indent=2))
# Forward the sampling parameters, seed and cycle marker through the worker launcher.
path=ROOT/'parallel_selfplay.py'; backup=path.with_name(path.name+'.pre-temp1-gui-20261002')
shutil.copy2(path,backup); text=path.read_text()
text=text.replace('    args = parser.parse_args()', '    parser.add_argument("--late-temperature", type=float, default=0.25)\n    parser.add_argument("--seed", type=int)\n    parser.add_argument("--experiment-cycle", type=int, default=-1)\n    args = parser.parse_args()')
text=text.replace('            process = subprocess.Popen(command)', '            command += ["--late-temperature", str(args.late_temperature), "--experiment-cycle", str(args.experiment_cycle)]\n            if args.seed is not None:\n                command += ["--seed", str(args.seed + game_number)]\n            process = subprocess.Popen(command)')
path.write_text(text)
path=ROOT/'selfplay.py'; shutil.copy2(path,path.with_name(path.name+'.pre-temp1-gui-20261002'))
text=path.read_text().replace('    args = parser.parse_args()', '    parser.add_argument("--experiment-cycle", type=int, default=-1)\n    args = parser.parse_args()')
text=text.replace('            late_temperature=np.asarray(args.late_temperature),','            late_temperature=np.asarray(args.late_temperature),\n            experiment_cycle=np.asarray(args.experiment_cycle),\n            seed=np.asarray(-1 if args.seed is None else args.seed),')
path.write_text(text)
shutil.copy2(WORK/'monitor_experiment.py',ROOT/'monitor_experiment.py')

text=(WIN/'training_controller_v2_raw.py').read_text(encoding='utf-8')
text=text.replace('raw-visits-v1','live-lr0001-temp1-20261002').replace('training_controller_v2_raw','training_controller_v2_temp1')
text=text.replace('v2 訪問回数版・独立学習','v2 低学習率・温度1の学習')
text=text.replace('119サイクルモデルのコピーから開始します','前回の最終モデルから継続（学習率0.0001・温度1）')
text=text.replace('旧データと旧チェックポイントから分離。確認対局の後、学習を1サイクルずつ実行できます。','前回の最終モデル・学習状態・80局を引き継ぎます。学習率0.0001、温度1、上限512半手。')
text=text.replace('state.get("cycles_per_run", 0) if state.get("schema_version") == 2 else 0','state.get("cycles_per_run", 10)')
text=text.replace('state.get("save_interval", 25)','state.get("save_interval", 1)')
text=text.replace("[/]run_live.py'","[/]run_live.py|[/]monitor_experiment.py'")
text=text.replace('             "--checkpoint", CHECKPOINT, "--output-dir", DATA],','             "--checkpoint", CHECKPOINT, "--output-dir", DATA,\n             "--late-temperature", "1", "--seed", str(20281002 + (self.cycles + 1) * 1000),\n             "--experiment-cycle", str(self.cycles + 1)],')
text=text.replace('                         "--training-state", TRAINING_STATE],','                         "--training-state", TRAINING_STATE,\n                         "--learning-rate", "0.0001", "--constant-learning-rate",\n                         "--seed", str(20281002 + cycle)],')
text=text.replace('                    if cycle % save_interval == 0:', '                    if cycle % save_interval == 0 or mode != "continuous" or (cycles_per_run and completed + 1 >= cycles_per_run):')
text=text.replace('["cp", CHECKPOINT, f"{SAVED}/cycle-{cycle:06d}.msgpack"]','["cp", "--no-clobber", CHECKPOINT, f"{SAVED}/cycle-{cycle:06d}.msgpack"]')
needle='                    completed += 1\n                    self.save_state()'
text=text.replace(needle, needle+'\n                    self.run_stage(\n                        "価値予測・反復・終局理由を確認中（10%以上の不活性で停止）",\n                        [PYTHON, "-u", "monitor_experiment.py", "--experiment", EXPERIMENT, "--cycle", str(cycle)],\n                    )')
text=text.replace("'[r]aw-visits-v1'", "'[l]ive-lr0001-temp1-20261002'")
# Verification games go to a separate directory and never enter training replay.
text=text.replace('    def generate(self, games: int, workers: int, simulations: int) -> None:', '    def generate(self, games: int, workers: int, simulations: int, verify: bool = False) -> None:\n        data_dir = f"{EXPERIMENT}/validation_data" if verify else DATA')
text=text.replace('"--output-dir", DATA,','"--output-dir", data_dir,').replace('"--data-dir", DATA,\n             "--min-games"','"--data-dir", data_dir,\n             "--min-games"')
text=text.replace('        self.games_total += games','        if not verify:\n            self.games_total += games')
text=text.replace('self.generate(2, 1, simulations)','self.generate(2, 1, simulations, verify=True)')
new_gui=WIN/'training_controller_v2_temp1.py'
if new_gui.exists(): raise RuntimeError('GUI already exists')
new_gui.write_text(text,encoding='utf-8')
(WIN/'v2温度1学習を起動.bat').write_text('@echo off\r\nstart "" "%~dp0.venv\\Scripts\\pythonw.exe" "%~dp0training_controller_v2_temp1.py"\r\n',encoding='utf-8')
(WIN/'training_controller_v2_temp1_state.json').write_text(json.dumps(dict(schema_version=2,cycles=0,games_total=0,games_per_cycle=8,workers=2,simulations=50,steps=64,cycles_per_run=10,save_interval=1),indent=2))
path=WIN/'model_comparison.py'; shutil.copy2(path,path.with_name(path.name+'.pre-temp1-gui-20261002'))
text=path.read_text(encoding='utf-8').replace('NO_WINDOW =', 'TEMP1_SAVED_MODELS = f"{PROJECT}/experiments/live-lr0001-temp1-20261002/checkpoints/saved"\nNO_WINDOW =',1)
text=text.replace('("低学習率・自己対局", LIVE_SAVED_MODELS))','("低学習率・自己対局", LIVE_SAVED_MODELS), ("低学習率・温度1", TEMP1_SAVED_MODELS))')
text=text.replace('{RAW_SAVED_MODELS} {LIVE_SAVED_MODELS} -maxdepth','{RAW_SAVED_MODELS} {LIVE_SAVED_MODELS} {TEMP1_SAVED_MODELS} -maxdepth')
text=text.replace("[/]run_live.py'","[/]run_live.py|[/]monitor_experiment.py'")
path.write_text(text,encoding='utf-8')
for name in ['training_controller_v2_raw.py','training_controller_v2.py']:
    path=WIN/name
    if path.exists():
        text=path.read_text(encoding='utf-8')
        if '[/]monitor_experiment.py' not in text:
            shutil.copy2(path,path.with_name(path.name+'.pre-temp1-gui-20261002'))
            path.write_text(text.replace("[/]run_live.py'","[/]run_live.py|[/]monitor_experiment.py'"),encoding='utf-8')
(DEST/'code').mkdir()
for name in ['selfplay.py','parallel_selfplay.py','train.py','monitor_experiment.py']:
    shutil.copy2(ROOT/name,DEST/'code'/name)
shutil.copy2(new_gui,WORK/'training_controller_v2_temp1.py')
shutil.copy2(DEST/'configuration.json',WORK/'configuration.json')
print(json.dumps(dict(prepared=str(DEST),carried_games=80,launcher=str(WIN/'v2温度1学習を起動.bat'),source_checkpoint_unchanged=digest(SOURCE/'checkpoints/latest.msgpack')==config['source_checkpoint_sha256'])))
