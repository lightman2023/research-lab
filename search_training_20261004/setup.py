import json
import shutil
import hashlib
from pathlib import Path

ROOT=Path('/home/user/chess/alphazero_chess_v2')
WORK=Path('/mnt/c/Users/User/Documents/ChatGPT/研究室/search_training_20261004')
WIN=Path('/mnt/c/Users/User/Desktop/chess_v2')
SOURCE=ROOT/'experiments/live-lr0001-temp1-20261002'
DEST=ROOT/'experiments/search-budget-20261004'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not DEST.exists(),DEST
assert digest(SOURCE/'checkpoints/latest.msgpack')==digest(SOURCE/'checkpoints/saved/cycle-000010.msgpack')
import sys
sys.path.insert(0,str(ROOT))
import chess.pgn
from flax import serialization
assert int(serialization.msgpack_restore((SOURCE/'checkpoints/training_state.msgpack').read_bytes())['step'])==2880
files=sorted((SOURCE/'selfplay_data').glob('*.npz'))
assert len(files)==160,len(files)
DEST.mkdir()
(DEST/'source').mkdir()
for name in ('latest.msgpack','training_state.msgpack'):
    shutil.copy2(SOURCE/'checkpoints'/name,DEST/'source'/name)
probe_source=ROOT/'arena_results/match-20261002-233038-c11626a7.pgn'
shutil.copy2(probe_source,DEST/'source/repetition_positions.pgn')
positions=[]
with probe_source.open() as f:
    for number in (1,2,3):
        game=chess.pgn.read_game(f)
        moves=list(game.mainline_moves())
        for offset in (5,1):
            ply=len(moves)-offset
            positions.append(dict(game=number,ply=ply,moves=[m.uci() for m in moves[:ply]],reference_move=moves[ply].uci()))
(DEST/'repetition_manifest.json').write_text(json.dumps(positions,indent=2))
configuration=dict(source_experiment=str(SOURCE),source_cycle=10,starting_training_step=2880,
    checkpoint_sha256=digest(DEST/'source/latest.msgpack'),state_sha256=digest(DEST/'source/training_state.msgpack'),
    carried_games=160,branch_names=['A50','B200'],simulations={'A50':50,'B200':200},
    learning_rate=.0001,constant_learning_rate=True,temperature=1,root_noise=True,
    games_per_cycle=8,workers=2,updates_per_cycle=64,batch_size=64,max_plies=512,
    target_cycles=5,initial_gate_cycles=1,save_interval=1,seed_base=20401004,
    evaluation_simulations=200,evaluation_games=24,policy_target='raw_mcts_visits_v1',
    draw_rule='az_actual_threefold_50move_v1',
    note='A and B inherit identical 160 training games and optimizer. New games differ; same seeds do not imply identical replay batches. Two prior cap draws remain inherited, separately counted. Evaluation data excluded.')
manifest=[]
for number,p in enumerate(files):
    manifest.append(dict(source=str(p),name=f'carried-{number:04d}-{p.name}',sha256=digest(p)))
for branch,simulations in [('A50',50),('B200',200)]:
    dest=DEST/branch
    (dest/'checkpoints/saved').mkdir(parents=True)
    (dest/'selfplay_data').mkdir()
    for name in ('latest.msgpack','training_state.msgpack'):
        shutil.copy2(DEST/'source'/name,dest/'checkpoints'/name)
    shutil.copy2(DEST/'source/latest.msgpack',dest/'checkpoints/saved/cycle-000000.msgpack')
    shutil.copy2(SOURCE/'evaluation_manifest.json',dest/'evaluation_manifest.json')
    shutil.copy2(DEST/'repetition_manifest.json',dest/'repetition_manifest.json')
    for p,entry in zip(files,manifest):
        shutil.copy2(p,dest/'selfplay_data'/entry['name'])
        assert digest(dest/'selfplay_data'/entry['name'])==entry['sha256']
    (dest/'configuration.json').write_text(json.dumps({**configuration,'branch':branch,'simulations_this_branch':simulations},indent=2))
    original=(WIN/'training_controller_v2_temp1.py').read_text(encoding='utf-8')
    text=original.replace('live-lr0001-temp1-20261002',f'search-budget-20261004/{branch}')
    text=text.replace('training_controller_v2_temp1',f'training_controller_search_{branch}')
    text=text.replace('v2 低学習率・温度1の学習',f'探索学習比較 {branch}：{simulations}回')
    text=text.replace('前回の最終モデル・学習状態・80局を引き継ぎます。学習率0.0001、温度1、上限512半手。',f'共通cycle 10と160局から開始。{simulations}回探索、学習率0.0001、温度1。初回1→計5サイクル。')
    text=text.replace('20281002','20401004')
    text=text.replace('"monitor_experiment.py", "--experiment"','"monitor_search_training.py", "--experiment"')
    text=text.replace('[/]monitor_experiment.py','[/]monitor_experiment.py|[/]monitor_search_training.py')
    needle='        return games, workers, simulations, steps, cycles_per_run, save_interval'
    assert text.count(needle)==1
    text=text.replace(needle,f'        if games != 8 or simulations != {simulations} or steps != 64:\n            raise ValueError("比較条件：自己対局8局・探索{simulations}回・更新64を維持してください。")\n'+needle)
    needle='        games, workers, simulations, steps, cycles_per_run, save_interval = settings\n        try:'
    assert text.count(needle)==1
    text=text.replace(needle,'        games, workers, simulations, steps, cycles_per_run, save_interval = settings\n        if mode != "verify":\n            if self.cycles >= 5:\n                self.root.after(0, lambda: self.status_var.set("予定の5サイクルは完了しています。"))\n                self.root.after(0, lambda: self.set_running(False))\n                return\n            # First run stops at one cycle; subsequent finite runs stop at five in total.\n            cycles_per_run = 1 if self.cycles == 0 else min(cycles_per_run or 5, 5 - self.cycles)\n        try:')
    text=text.replace('while mode != "continuous" or cycles_per_run == 0 or completed < cycles_per_run:','while self.cycles < 5 and (mode != "continuous" or completed < cycles_per_run):')
    text=text.replace('f"新方式の学習 {self.cycles}サイクルまで完了"','f"学習 {self.cycles}/5サイクル完了。初回1のログを確認後、連続学習で5まで進めます。"')
    text=text.replace("'[l]search-budget-20261004/", "'[s]earch-budget-20261004/") if False else text
    # stop regex inherited [l]ive... must point to this branch only.
    text=text.replace(f"'[l]ive-lr0001-temp1-20261002'",f"'[s]earch-budget-20261004/{branch}'")
    text=text.replace(f"'[l]search-budget-20261004/{branch}'",f"'[s]earch-budget-20261004/{branch}'")
    controller=WIN/f'training_controller_search_{branch}.py'
    assert not controller.exists()
    controller.write_text(text,encoding='utf-8')
    shutil.copy2(controller,WORK/controller.name)
    (WIN/f'training_controller_search_{branch}_state.json').write_text(json.dumps(dict(schema_version=2,cycles=0,games_total=0,games_per_cycle=8,workers=2,simulations=simulations,steps=64,cycles_per_run=4,save_interval=1),indent=2))
    (WIN/f'探索学習比較_{branch}を起動.bat').write_text(f'@echo off\nstart "" "%~dp0.venv\\Scripts\\pythonw.exe" "%~dp0{controller.name}"\n',encoding='cp932')
shutil.copy2(WORK/'monitor_search_training.py',ROOT/'monitor_search_training.py')
(DEST/'configuration.json').write_text(json.dumps(configuration,indent=2))
(DEST/'carried_data_manifest.json').write_text(json.dumps(manifest,indent=2))
for name in ('configuration.json','carried_data_manifest.json'):
    shutil.copy2(DEST/name,WORK/name)
(DEST/'code').mkdir()
for name in ('train.py','parallel_selfplay.py','selfplay.py','monitor_experiment.py','monitor_search_training.py'):
    shutil.copy2(ROOT/name,DEST/'code'/name)
assert digest(SOURCE/'checkpoints/latest.msgpack')==configuration['checkpoint_sha256']
assert digest(SOURCE/'checkpoints/training_state.msgpack')==configuration['state_sha256']
print(json.dumps(dict(prepared=str(DEST),source_unchanged=True,carried_games_each=160,starting_step=2880)))
