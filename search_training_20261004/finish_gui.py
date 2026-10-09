from pathlib import Path
import shutil

win=Path('C:/Users/User/Desktop/chess_v2')
path=win/'model_comparison.py'
text=path.read_text(encoding='utf-8')
shutil.copy2(path,path.with_name(path.name+'.pre-budget-training-20261004'))
needle='NO_WINDOW = subprocess.CREATE_NO_WINDOW'
assert text.count(needle)==1
text=text.replace(needle,'SEARCH_A_SAVED_MODELS = f"{PROJECT}/experiments/search-budget-20261004/A50/checkpoints/saved"\nSEARCH_B_SAVED_MODELS = f"{PROJECT}/experiments/search-budget-20261004/B200/checkpoints/saved"\n'+needle)
needle='("低学習率・温度1", TEMP1_SAVED_MODELS))'
assert text.count(needle)==1
text=text.replace(needle,'("低学習率・温度1", TEMP1_SAVED_MODELS), ("探索学習A50", SEARCH_A_SAVED_MODELS), ("探索学習B200", SEARCH_B_SAVED_MODELS))')
text=text.replace('{TEMP1_SAVED_MODELS} -maxdepth','{TEMP1_SAVED_MODELS} {SEARCH_A_SAVED_MODELS} {SEARCH_B_SAVED_MODELS} -maxdepth')
text=text.replace('[/]monitor_experiment.py','[/]monitor_experiment.py|[/]monitor_search_training.py')
path.write_text(text,encoding='utf-8')
text=(win/'model_comparison_search.py').read_text(encoding='utf-8')
text=text.replace('SearchComparisonApp','TrainedSearchComparisonApp')
text=text.replace('温度1 cycle 10：探索50回対200回','探索回数別の学習後比較：両側200回探索')
text=text.replace('        self.simulations_b_var.set("200")','        self.simulations_var.set("200")\n        self.simulations_b_var.set("200")')
text=text.replace('a = "低学習率・温度1 / cycle-000010.msgpack"','a = "探索学習A50 / cycle-000005.msgpack"')
text=text.replace('b = "低学習率・温度1 / cycle-000010.msgpack"','b = "探索学習B200 / cycle-000005.msgpack"')
text=text.replace('温度1のcycle 10が見つかりません。一覧を確認してください。','両条件の5サイクル終了モデルがまだありません。学習完了後に一覧を更新してください。')
text=text.replace('同じcycle 10：A探索50回、B探索200回。24局・上限512半手。','A＝50探索で学習、B＝200探索で学習。評価は両側200回・24局・上限512半手。')
text=text.replace('温度1cycle 10同士／24局／A探索50回・B探索200回／上限512半手','探索回数別学習cycle 5同士／24局／評価は両側200回／上限512半手')
(win/'model_comparison_trained_search.py').write_text(text,encoding='utf-8')
(win/'探索学習後のモデルを比較.bat').write_text('@echo off\nstart "" "%~dp0.venv\\Scripts\\pythonw.exe" "%~dp0model_comparison_trained_search.py"\n',encoding='cp932')
print('Comparison sources and final evaluation launcher added.')
