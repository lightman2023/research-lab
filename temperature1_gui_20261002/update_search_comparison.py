from pathlib import Path

gui_dir=Path('C:/Users/User/Desktop/chess_v2')
project=Path('//wsl.localhost/Ubuntu/home/user/chess/alphazero_chess_v2')

def edit(path, replacements):
    text=path.read_text(encoding='utf-8')
    original=text
    for old,new in replacements:
        assert text.count(old)==1, (path,old[:100],text.count(old))
        text=text.replace(old,new)
    backup=path.with_name(path.name+'.pre-search-comparison-20261003')
    if not backup.exists(): backup.write_text(original,encoding='utf-8')
    path.write_text(text,encoding='utf-8')

edit(gui_dir/'model_comparison.py',[
('self.simulations_var = tk.StringVar(value="50")','self.simulations_var = tk.StringVar(value="50")\n        self.simulations_b_var = tk.StringVar(value="50")'),
('学習による強さの変化を測ります。','モデルや探索回数の違いを測ります。'),
('text="　1手のシミュレーション："','text="　A探索："'),
('        ttk.Label(options, text="　最大手数：").pack(side="left")','        ttk.Label(options, text="　B探索：").pack(side="left")\n        self.sims_b_spin = ttk.Spinbox(\n            options, from_=1, to=1000, width=6, textvariable=self.simulations_b_var\n        )\n        self.sims_b_spin.pack(side="left")\n        ttk.Label(options, text="　上限（半手）：").pack(side="left")'),
('        if self.model_paths[model_a] == self.model_paths[model_b]:\n            messagebox.showwarning("同じモデル", "異なる学習回数のモデルを選んでください。")\n            return\n',''),
('            max_plies = self.parse_positive_int(self.max_plies_var.get(), "最大手数", 20)','            simulations_b = self.parse_positive_int(\n                self.simulations_b_var.get(), "Bのシミュレーション数"\n            )\n            max_plies = self.parse_positive_int(self.max_plies_var.get(), "最大手数", 20)'),
('        if games % 2:','        if self.model_paths[model_a] == self.model_paths[model_b] and simulations == simulations_b:\n            messagebox.showwarning("同じ条件", "同じモデルを比較する場合は、AとBの探索回数を変えてください。")\n            return\n        if games % 2:'),
('f"=== 比較開始：A={model_a} / B={model_b} / {games}局 / {simulations}シミュレーション ==="','f"=== 比較開始：A={model_a}（探索{simulations}回） / B={model_b}（探索{simulations_b}回） / {games}局 / 上限{max_plies}半手 ==="'),
('games, simulations, max_plies, self.match_id)','games, simulations, simulations_b, max_plies, self.match_id)'),
('        self.sims_spin.configure(state=spin_state)','        self.sims_spin.configure(state=spin_state)\n        self.sims_b_spin.configure(state=spin_state)'),
('        simulations: int,\n        max_plies: int,','        simulations: int,\n        simulations_b: int,\n        max_plies: int,'),
('f"--games {games} --simulations {simulations} --max-plies {max_plies} "','f"--games {games} --simulations-a {simulations} --simulations-b {simulations_b} --max-plies {max_plies} "'),
('elif event == "game_start":','elif event == "game_start":'),
('            self.append_log(f"棋譜保存先（WSL）：{data[\'pgn\']}")','            self.append_log(f"棋譜保存先（WSL）：{data[\'pgn\']}")\n            if data.get("metadata"):\n                self.append_log(f"設定・結果保存先（WSL）：{data[\'metadata\']}")'),
])

edit(project/'arena.py',[
('import argparse','import argparse\nimport hashlib'),
('    parser.add_argument("--simulations", type=int, default=50)','    parser.add_argument("--simulations", type=int, default=50)\n    parser.add_argument("--simulations-a", type=int)\n    parser.add_argument("--simulations-b", type=int)'),
('    if args.simulations < 1:\n        parser.error("--simulations must be at least 1")','    simulations_a = args.simulations if args.simulations_a is None else args.simulations_a\n    simulations_b = args.simulations if args.simulations_b is None else args.simulations_b\n    if min(simulations_a, simulations_b) < 1:\n        parser.error("simulations for A and B must be at least 1")'),
('    if args.model_a.resolve() == args.model_b.resolve():\n        parser.error("model A and model B must be different checkpoints")','    if args.model_a.resolve() == args.model_b.resolve() and simulations_a == simulations_b:\n        parser.error("same checkpoint requires different simulation counts")\n    label_a = f"A: {model_name(args.model_a)} [sims={simulations_a}]"\n    label_b = f"B: {model_name(args.model_b)} [sims={simulations_b}]"'),
('        simulations=args.simulations,','        simulations=args.simulations,\n        simulations_a=simulations_a,\n        simulations_b=simulations_b,'),
('    model_b, variables_b = load_variables(args.model_b)\n    agent_a = MCTS(model_a, variables_a, simulations=args.simulations)\n    agent_b = MCTS(model_b, variables_b, simulations=args.simulations)','    if args.model_a.resolve() == args.model_b.resolve():\n        model_b, variables_b = model_a, variables_a\n    else:\n        model_b, variables_b = load_variables(args.model_b)\n    agent_a = MCTS(model_a, variables_a, simulations=simulations_a)\n    agent_b = MCTS(model_b, variables_b, simulations=simulations_b)'),
('    a_wins = b_wins = draws = cutoffs = 0','    metadata_path = pgn_path.with_suffix(".json")\n    metadata = dict(model_a=str(args.model_a.resolve()), model_b=str(args.model_b.resolve()),\n        model_a_sha256=hashlib.sha256(args.model_a.read_bytes()).hexdigest(),\n        model_b_sha256=hashlib.sha256(args.model_b.read_bytes()).hexdigest(),\n        simulations_a=simulations_a, simulations_b=simulations_b,\n        games_requested=args.games, max_plies=args.max_plies, seed=args.seed,\n        draw_rule=DRAW_RULE, root_noise=False, temperature=0.0, results=[])\n    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")\n    a_wins = b_wins = draws = cutoffs = 0'),
('            game.headers["White"] = model_name(args.model_a if a_is_white else args.model_b)\n            game.headers["Black"] = model_name(args.model_b if a_is_white else args.model_a)','            game.headers["White"] = label_a if a_is_white else label_b\n            game.headers["Black"] = label_b if a_is_white else label_a\n            game.headers["WhiteSimulations"] = str(simulations_a if a_is_white else simulations_b)\n            game.headers["BlackSimulations"] = str(simulations_b if a_is_white else simulations_a)\n            game.headers["MaxPlies"] = str(args.max_plies)\n            game.headers["Seed"] = str(args.seed)'),
('            completed = game_index + 1','            metadata["results"].append(dict(game=game_index+1, a_is_white=a_is_white,\n                opening=opening_name, result=result, termination=termination,\n                plies=len(board.move_stack), winner=winner))\n            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")\n            completed = game_index + 1'),
('        pgn=str(pgn_path.resolve()),','        pgn=str(pgn_path.resolve()),\n        metadata=str(metadata_path.resolve()),\n        simulations_a=simulations_a,\n        simulations_b=simulations_b,'),
])

preset=(gui_dir/'model_comparison_temp1.py').read_text(encoding='utf-8')
preset=preset.replace('Temperature1ComparisonApp','SearchComparisonApp')
preset=preset.replace('cycle 0 and cycle 10','cycle 10 at 50 and 200 simulations')
preset=preset.replace('温度1 学習前後のモデル比較（0対10）','温度1 cycle 10：探索50回対200回')
preset=preset.replace('        self.max_plies_var.set("512")','        self.max_plies_var.set("512")\n        self.simulations_b_var.set("200")')
preset=preset.replace('cycle-000000.msgpack','cycle-000010.msgpack')
preset=preset.replace('温度1のcycle 0と10','温度1のcycle 10')
preset=preset.replace('A＝学習前、B＝学習後。24局・探索50回・上限512半手で比較します。','同じcycle 10：A探索50回、B探索200回。24局・上限512半手。')
preset=preset.replace('温度1のcycle 0対10／24局／探索50回／上限512半手','温度1cycle 10同士／24局／A探索50回・B探索200回／上限512半手')
target=gui_dir/'model_comparison_search.py'
assert not target.exists(), target
target.write_text(preset,encoding='utf-8')
(gui_dir/'v2探索50対200を比較.bat').write_text('@echo off\nstart "" "%~dp0.venv\\Scripts\\pythonw.exe" "%~dp0model_comparison_search.py"\n',encoding='cp932')
for path in (gui_dir/'温度1学習README.md',project/'README.md'):
    with path.open('a',encoding='utf-8') as f:
        f.write('\n\n## 同じモデルで探索回数を比較（2026-10-03）\n\n'
            'Windowsの `v2探索50対200を比較.bat` を開く。両側に温度1のcycle-000010を選択済み。'
            '初期条件はA探索50回、B探索200回、24局、上限512半手。比較対戦をスタートで開始する。'
            '既存GUIもA・B別の探索回数に対応。同じモデルは探索回数が異なる場合に比較できる。'
            '棋譜の白黒名にA/Bと探索数を記録し、白黒交換後も条件を識別できる。'
            '棋譜と同名のJSONにモデルのSHA256・探索数・seed・上限・各局の結果を保存する。'
            'CLIは `--simulations-a 50 --simulations-b 200` を指定する。従来の `--simulations 50` は両側50回として引き続き使える。'
            '今回の24局の本評価はまだ実行していない。\n')
print('Updated GUI, arena, launch preset, documentation; originals preserved.')
