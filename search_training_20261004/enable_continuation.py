import json
import shutil
from pathlib import Path
win=Path('C:/Users/User/Desktop/chess_v2')
work=Path(__file__).resolve().parent
path=win/'training_controller_search_B200.py'
text=path.read_text(encoding='utf-8')
shutil.copy2(path,path.with_name(path.name+'.pre-continuation-20261005'))
old='''        if mode != "verify":
            if self.cycles >= 5:
                self.root.after(0, lambda: self.status_var.set("予定の5サイクルは完了しています。"))
                self.root.after(0, lambda: self.set_running(False))
                return
            # First run stops at one cycle; subsequent finite runs stop at five in total.
            cycles_per_run = 1 if self.cycles == 0 else min(cycles_per_run or 5, 5 - self.cycles)
'''
assert text.count(old)==1
text=text.replace(old,'')
text=text.replace('while self.cycles < 5 and (mode != "continuous" or completed < cycles_per_run):','while mode != "continuous" or cycles_per_run == 0 or completed < cycles_per_run:')
text=text.replace('f"学習 {self.cycles}/5サイクル完了。初回1のログを確認後、連続学習で5まで進めます。"','f"Bの学習 {self.cycles}サイクルまで完了"')
text=text.replace('[t]rain[.]py|[a]rena[.]py|[r]un_live[.]py','[t]rain[.]py|[r]un_live[.]py')
text=text.replace('別の自己対局・学習・対戦が動いています。','別の自己対局・学習・監視が動いています。評価対局との同時実行は可能です。')
text=text.replace('        try:\n            settings = self.settings()','        if self.running:\n            return\n        try:\n            settings = self.settings()',1)
text=text.replace('共通cycle 10と160局から開始。200回探索、学習率0.0001、温度1。初回1→計5サイクル。','Bの最新状態から追加学習。200探索・学習率0.0001・温度1。回数0は手動停止まで。評価と同時実行可。')
text=text.replace('探索学習比較 B200：200回','B200 追加学習（評価と並行可能）')
text=text.replace('state.get("cycles_per_run", 10)','state.get("cycles_per_run", 0)')
old='''    Controller(root)
    root.mainloop()'''
new='''    controller = Controller(root)
    import sys
    if "--start-continuous" in sys.argv:
        root.after(500, controller.start_continuous)
    root.mainloop()'''
assert text.count(old)==1
text=text.replace(old,new)
path.write_text(text,encoding='utf-8')
shutil.copy2(path,work/'training_controller_search_B200_continuation.py')
state_path=win/'training_controller_search_B200_state.json'
state=json.loads(state_path.read_text(encoding='utf-8'))
assert state['cycles']==5 and state['games_total']==40
shutil.copy2(state_path,state_path.with_name(state_path.name+'.pre-continuation-20261005'))
state['cycles_per_run']=0
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
# Saved checkpoint paths are the only model sources in this GUI. Allow B200
# training while still blocking other training branches.
path=win/'model_comparison.py';text=path.read_text(encoding='utf-8')
shutil.copy2(path,path.with_name(path.name+'.pre-continuation-20261005'))
old='''        command = (
            "pgrep -f '[p]arallel_selfplay[.]py|[s]elfplay[.]py|[t]rain[.]py|[a]rena[.]py|[r]un_live[.]py|[m]onitor_experiment[.]py|[m]onitor_search_training[.]py|[c]heck_search_cycle[.]py' >/dev/null"
        )
        return self.run_wsl(command, check=False).returncode == 0'''
new='''        command = (
            "pgrep -af '[p]arallel_selfplay[.]py|[s]elfplay[.]py|[t]rain[.]py|[r]un_live[.]py|[m]onitor_experiment[.]py|[m]onitor_search_training[.]py|[c]heck_search_cycle[.]py'"
        )
        result = self.run_wsl(command, capture_output=True, check=False)
        if result.returncode not in (0, 1):
            return True
        allowed = f"{PROJECT}/experiments/search-budget-20261004/B200"
        return any(allowed not in line for line in result.stdout.splitlines() if line.strip())'''
assert text.count(old)==1
text=text.replace(old,new)
path.write_text(text,encoding='utf-8')
root=Path('//wsl.localhost/Ubuntu/home/user/chess/alphazero_chess_v2/experiments/search-budget-20261004')
protocol=dict(start_cycle=5,start_step=3200,cycles_per_run=0,
    save_interval=state['save_interval'],simulations=200,learning_rate=.0001,temperature=1,
    games_per_cycle=8,updates_per_cycle=64,batch_size=64,
    concurrent_evaluation_allowed=True,frozen_comparison_cycle=5,
    note='Initial paired 5-cycle experiment preserved. B-only extension; A remains at cycle 5. Original experiment results unchanged.')
(root/'B200/continuation_protocol_20261005.json').write_text(json.dumps(protocol,indent=2))
(work/'continuation_protocol_20261005.json').write_text(json.dumps(protocol,indent=2))
print('B continuation enabled; current comparison snapshots retained; default cycles=0.')
