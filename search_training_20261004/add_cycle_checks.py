from pathlib import Path
import shutil
root=Path('//wsl.localhost/Ubuntu/home/user/chess/alphazero_chess_v2')
work=Path(__file__).resolve().parent
win=Path('C:/Users/User/Desktop/chess_v2')
shutil.copy2(work/'check_search_cycle.py',root/'check_search_cycle.py')
for branch in ('A50','B200'):
    path=win/f'training_controller_search_{branch}.py'
    text=path.read_text(encoding='utf-8')
    needle='                    self.generate(games, workers, simulations)'
    assert text.count(needle)==1
    text=text.replace(needle,'                    self.run_stage("サイクル開始前のデータ・学習状態を照合中",\n                        [PYTHON, "-u", "check_search_cycle.py", "--experiment", EXPERIMENT, "--cycle", str(cycle), "--stage", "before"])\n'+needle+'\n                    self.run_stage("新しい8局の保存数を照合中",\n                        [PYTHON, "-u", "check_search_cycle.py", "--experiment", EXPERIMENT, "--cycle", str(cycle), "--stage", "after"])')
    path.write_text(text,encoding='utf-8')
    shutil.copy2(path,work/path.name)
dest=root/'experiments/search-budget-20261004'
shutil.copy2(work/'check_search_cycle.py',dest/'code/check_search_cycle.py')
shutil.copytree(root/'azchess',dest/'code/azchess',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(work/'README.md',dest/'README.md')
with (root/'README.md').open('a',encoding='utf-8') as f:
    f.write('\n\n## 自己対局探索予算の学習比較（2026-10-04）\n\n'
        '独立実験はexperiments/search-budget-20261004のA50/B200。温度1cycle 10・160局・2880更新から同条件で分岐。'
        'Windowsの探索学習比較_A50/B200を起動.batでまず各1サイクル、その後累計5まで実行。'
        '8局・64更新・一定学習率0.0001・温度1を維持し、探索数のみ50/200。'
        '学習後は探索学習後のモデルを比較.batで両側200探索・24局。各実験README.mdに操作・監視・記録を記載。'
        '中断状態では対局数・学習状態の照合で止まることがある。既存データを消さず、同サイクルの重複生成を避ける。\n')
print('Cycle checks and reproducibility snapshots saved.')
