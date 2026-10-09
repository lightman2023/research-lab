from pathlib import Path
import shutil
win=Path('C:/Users/User/Desktop/chess_v2')
root=Path('//wsl.localhost/Ubuntu/home/user/chess/alphazero_chess_v2')
def edit(path,replacements):
    text=path.read_text(encoding='utf-8-sig')
    backup=path.with_name(path.name+'.pre-cpu-evaluation-20261005')
    if not backup.exists():shutil.copy2(path,backup)
    for old,new in replacements:
        assert text.count(old)==1,(path,old[:70],text.count(old))
        text=text.replace(old,new)
    path.write_text(text,encoding='utf-8')
edit(root/'arena.py',[
('import argparse','import argparse\nimport os'),
('from azchess.checkpoint import load_variables\nfrom azchess.mcts import MCTS\n',''),
('    args = parser.parse_args()','    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")\n    args = parser.parse_args()\n    if args.device == "cpu":\n        os.environ["JAX_PLATFORMS"] = "cpu"\n    from azchess.checkpoint import load_variables\n    from azchess.mcts import MCTS\n    import jax\n    actual_devices = [str(device) for device in jax.devices()]\n    if args.device == "cpu" and any(device.platform != "cpu" for device in jax.devices()):\n        raise RuntimeError("CPU evaluation requested, but a non-CPU device was initialized")'),
('        simulations=args.simulations,\n        simulations_a=simulations_a,\n        simulations_b=simulations_b,','        simulations=args.simulations,\n        simulations_a=simulations_a,\n        simulations_b=simulations_b,\n        device=args.device,\n        actual_devices=actual_devices,'),
('        draw_rule=DRAW_RULE, root_noise=False, temperature=0.0, results=[])','        draw_rule=DRAW_RULE, root_noise=False, temperature=0.0,\n        device=args.device, actual_devices=actual_devices, results=[])'),
('            game.headers["Seed"] = str(args.seed)','            game.headers["Seed"] = str(args.seed)\n            game.headers["Device"] = args.device'),
('        metadata=str(metadata_path.resolve()),','        metadata=str(metadata_path.resolve()),\n        device=args.device,'),
])
edit(win/'model_comparison.py',[
('        self.max_plies_var = tk.StringVar(value="300")','        self.max_plies_var = tk.StringVar(value="300")\n        self.device_var = tk.StringVar(value="cpu")'),
('        buttons = ttk.Frame(frame)','        device_row = ttk.Frame(frame)\n        device_row.pack(fill="x", pady=(6, 0))\n        ttk.Label(device_row, text="比較の実行先：").pack(side="left")\n        self.device_combo = ttk.Combobox(device_row, textvariable=self.device_var,\n            values=("cpu", "gpu"), state="readonly", width=8)\n        self.device_combo.pack(side="left")\n        ttk.Label(device_row, text="CPUならGPU学習と競合しません。両モデルは同じ実行先で評価します。").pack(side="left", padx=8)\n\n        buttons = ttk.Frame(frame)'),
('f"=== 比較開始：A={model_a}（探索{simulations}回） / B={model_b}（探索{simulations_b}回） / {games}局 / 上限{max_plies}半手 ==="','f"=== 比較開始：A={model_a}（探索{simulations}回） / B={model_b}（探索{simulations_b}回） / {games}局 / 上限{max_plies}半手 / 実行先={self.device_var.get()} ==="'),
('        self.plies_spin.configure(state=spin_state)','        self.plies_spin.configure(state=spin_state)\n        self.device_combo.configure(state=combo_state)'),
('            f"--match-id {match_id}"','            f"--match-id {match_id} --device {self.device_var.get()}"'),
('            self.status_var.set("モデルをGPUへ読み込んでいます。初回は少し時間がかかります…")','            self.status_var.set(f"モデルを{data.get(\'device\', \'gpu\').upper()}へ読み込んでいます…")'),
('        message = last_output or f"終了コード {exit_code}"','        message = f"終了コード {exit_code}\\n" + (last_output or "出力なし")'),
])
print('CPU evaluation option and explicit exit-code logging added.')
