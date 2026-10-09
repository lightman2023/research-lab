"""Preset for comparing temperature-1 training cycle 10 at 50 and 200 simulations."""
import tkinter as tk
from model_comparison import ModelComparisonApp


class TrainedSearchComparisonApp(ModelComparisonApp):
    def __init__(self, root):
        self.preset_applied = False
        super().__init__(root)
        self.root.title("探索回数別の学習後比較：両側200回探索")
        self.max_plies_var.set("512")
        self.simulations_var.set("200")
        self.simulations_b_var.set("200")

    def show_models(self, names, paths, returncode):
        super().show_models(names, paths, returncode)
        if returncode != 0 or self.preset_applied:
            return
        a = "探索学習A50 / cycle-000005.msgpack"
        b = "探索学習B200 / cycle-000005.msgpack"
        if a not in paths or b not in paths:
            self.model_a_var.set("")
            self.model_b_var.set("")
            self.status_var.set("両条件の5サイクル終了モデルがまだありません。学習完了後に一覧を更新してください。")
            return
        self.model_a_var.set(a)
        self.model_b_var.set(b)
        self.preset_applied = True
        self.status_var.set("A＝50探索で学習、B＝200探索で学習。評価は両側200回・24局・上限512半手。")
        self.append_log("比較設定：探索回数別学習cycle 5同士／24局／評価は両側200回／上限512半手")


if __name__ == "__main__":
    root = tk.Tk()
    TrainedSearchComparisonApp(root)
    root.mainloop()
