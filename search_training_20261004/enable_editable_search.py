from pathlib import Path
import shutil
work=Path(__file__).resolve().parent
gui=Path('C:/Users/User/Desktop/chess_v2/training_controller_search_B200.py')
project=Path('//wsl.localhost/Ubuntu/home/user/chess/alphazero_chess_v2')
for path in [gui,project/'check_search_cycle.py']:
    backup=path.with_name(path.name+'.pre-editable-search-20261008')
    if not backup.exists():shutil.copy2(path,backup)
text=gui.read_text(encoding='utf-8')
text=text.replace('B200 追加学習（評価と並行可能）','B 追加学習（探索回数を変更可能）')
text=text.replace('Bの最新状態から追加学習。200探索・学習率0.0001・温度1。回数0は手動停止まで。評価と同時実行可。','探索回数は変更可能。途中のサイクルは元の回数で完了し、次から変更を適用。連続回数0は停止まで。')
text=text.replace('("一手の探索", self.simulations_var, 1, 800)','("一手の探索", self.simulations_var, 1, 10000)')
text=text.replace('1 <= simulations <= 800','1 <= simulations <= 10000')
text=text.replace('if games != 8 or simulations != 200 or steps != 64:', 'if games != 8 or steps != 64:')
text=text.replace('比較条件：自己対局8局・探索200回・更新64を維持してください。','1サイクルの自己対局8局・学習更新64を指定してください。探索回数は1～10000で変更できます。')
text=text.replace('"before", "--resume-partial"]','"before", "--resume-partial", "--simulations", str(simulations)]')
old='''                    self.cycle_plan = json.loads(planning)
                    self.generate(games, workers, simulations)'''
new='''                    self.cycle_plan = json.loads(planning)
                    cycle_simulations = self.cycle_plan["simulations_for_cycle"]
                    if cycle_simulations != simulations:
                        self.root.after(0, lambda old=cycle_simulations, new=simulations: self.append_log(
                            f"途中サイクルは{old}回探索で完了します。次のサイクルから{new}回を適用します。"))
                    self.generate(games, workers, cycle_simulations)'''
assert old in text;text=text.replace(old,new)
text=text.replace('"--stage", "after"]','"--stage", "after", "--simulations", str(cycle_simulations)]')
gui.write_text(text,encoding='utf-8')
shutil.copy2(gui,work/'training_controller_search_B200_continuation.py')
for name in ['check_search_cycle.py','cycle_search_settings.py']:shutil.copy2(work/name,project/name)
print('Updated editable search budget and persisted per-cycle settings; models and data untouched.')
