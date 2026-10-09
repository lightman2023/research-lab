from pathlib import Path

app_dir = Path('C:/Users/User/Desktop/chess_v2')
source = '''"""Preset for comparing temperature-1 training cycle 0 and cycle 10."""
import tkinter as tk
from model_comparison import ModelComparisonApp


class Temperature1ComparisonApp(ModelComparisonApp):
    def __init__(self, root):
        self.preset_applied = False
        super().__init__(root)
        self.root.title("温度1 学習前後のモデル比較（0対10）")
        self.max_plies_var.set("512")

    def show_models(self, names, paths, returncode):
        super().show_models(names, paths, returncode)
        if returncode != 0 or self.preset_applied:
            return
        a = "低学習率・温度1 / cycle-000000.msgpack"
        b = "低学習率・温度1 / cycle-000010.msgpack"
        if a not in paths or b not in paths:
            self.model_a_var.set("")
            self.model_b_var.set("")
            self.status_var.set("温度1のcycle 0と10が見つかりません。一覧を確認してください。")
            return
        self.model_a_var.set(a)
        self.model_b_var.set(b)
        self.preset_applied = True
        self.status_var.set("A＝学習前、B＝学習後。24局・探索50回・上限512半手で比較します。")
        self.append_log("比較設定：温度1のcycle 0対10／24局／探索50回／上限512半手")


if __name__ == "__main__":
    root = tk.Tk()
    Temperature1ComparisonApp(root)
    root.mainloop()
'''
target = app_dir / 'model_comparison_temp1.py'
if target.exists() and target.read_text(encoding='utf-8') != source:
    raise RuntimeError('Existing comparison preset differs; preserve and inspect it first.')
target.write_text(source, encoding='utf-8')
(app_dir / 'v2温度1の学習前後を比較.bat').write_text(
    '@echo off\nstart "" "%~dp0.venv\\Scripts\\pythonw.exe" "%~dp0model_comparison_temp1.py"\n',
    encoding='cp932')
readme = app_dir / '温度1学習README.md'
text = readme.read_text(encoding='utf-8')
text = text.replace('Windows側のGUI初期化を確認。10サイクルの本学習はまだ実行していない。',
                    'Windows側のGUI初期化を確認。10サイクルの本学習は2026-10-02 23:20に完了。')
section = '''
## 温度1の学習前後を選んだ比較入口

`v2温度1の学習前後を比較.bat` を開き、「比較対戦をスタート」を押す。
既存の比較GUIを使い、Aに温度1のcycle-000000、Bにcycle-000010を自動選択する。
初期条件は24局（12開始局面で白黒交換）、一手50探索、上限512半手。
自己対局時の温度1と異なり、評価では訪問回数最多の手を選ぶ。
勝敗・終局理由・半手数・打ち切りを画面内ログで確認できる。
ログはmodel_comparison_v2.log、棋譜は完了時に表示されるWSLのパスに保存される。
今回の学習前後比較はまだ実行していない。打ち切りや引き分けが多い場合は強さを断定しない。
'''
if '## 温度1の学習前後を選んだ比較入口' not in text:
    text += section
readme.write_text(text, encoding='utf-8')
print(target)
