"""Preset for comparing temperature-1 training cycle 10 at 50 and 200 simulations."""
import tkinter as tk
from model_comparison import ModelComparisonApp


class SearchComparisonApp(ModelComparisonApp):
    def __init__(self, root):
        self.preset_applied = False
        super().__init__(root)
        self.root.title("温度1 cycle 10：探索50回対200回")
        self.max_plies_var.set("512")
        self.simulations_b_var.set("200")

    def show_models(self, names, paths, returncode):
        super().show_models(names, paths, returncode)
        if returncode != 0 or self.preset_applied:
            return
        a = "低学習率・温度1 / cycle-000010.msgpack"
        b = "低学習率・温度1 / cycle-000010.msgpack"
        if a not in paths or b not in paths:
            self.model_a_var.set("")
            self.model_b_var.set("")
            self.status_var.set("温度1のcycle 10が見つかりません。一覧を確認してください。")
            return
        self.model_a_var.set(a)
        self.model_b_var.set(b)
        self.preset_applied = True
        self.status_var.set("同じcycle 10：A探索50回、B探索200回。24局・上限512半手。")
        self.append_log("比較設定：温度1cycle 10同士／24局／A探索50回・B探索200回／上限512半手")


if __name__ == "__main__":
    root = tk.Tk()
    SearchComparisonApp(root)
    root.mainloop()
