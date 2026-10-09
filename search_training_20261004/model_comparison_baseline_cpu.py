"""Frozen starting model versus B cycle 5, CPU evaluation beside GPU training."""
import tkinter as tk
from model_comparison import ModelComparisonApp

class BaselineCPUApp(ModelComparisonApp):
    def __init__(self,root):
        self.preset_applied=False
        super().__init__(root)
        root.title('開始モデル対B学習後：CPU比較／GPU学習と並行')
        self.games_var.set('12');self.simulations_var.set('200')
        self.simulations_b_var.set('200');self.max_plies_var.set('512');self.device_var.set('cpu')

    def show_models(self,names,paths,returncode):
        super().show_models(names,paths,returncode)
        if returncode or self.preset_applied:return
        a='低学習率・温度1 / cycle-000010.msgpack'
        b='探索学習B200 / cycle-000005.msgpack'
        if a not in paths or b not in paths:
            self.model_a_var.set('');self.model_b_var.set('');self.status_var.set('比較モデルがありません。');return
        self.model_a_var.set(a);self.model_b_var.set(b);self.preset_applied=True
        self.status_var.set('開始モデル対Bのcycle 5：両側200探索・12局・CPU。保存モデル固定。')
        self.append_log('開始モデル対B cycle 5のCPU比較を準備。GPU比較の失敗棋譜は保持。')
        import sys
        if '--start' in sys.argv:self.root.after(200,self.start_match)

if __name__=='__main__':
    root=tk.Tk();BaselineCPUApp(root);root.mainloop()
