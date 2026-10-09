"""Preset for comparing temperature-1 training cycle 0 and cycle 10."""
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
