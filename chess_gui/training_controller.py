"""AlphaZero型チェスAIの連続強化学習を管理するWindows GUI。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk


PROJECT = "/home/user/chess/alphazero_chess"
PYTHON = f"{PROJECT}/.venv/bin/python"
SAVED_MODELS = f"{PROJECT}/checkpoints/saved"
NO_WINDOW = subprocess.CREATE_NO_WINDOW
APP_DIR = Path(__file__).resolve().parent
LOG_FILE = APP_DIR / "training_controller.log"
STATE_FILE = APP_DIR / "training_controller_state.json"
MODEL_NAME = re.compile(r"^run-(\d{6,})\.msgpack$")
GAME_RESULT = re.compile(r"^Game \d+: (1-0|0-1|1/2-1/2|\*)(?:\s|$)")
CUTOFF_SCORE = re.compile(r"termination=MAX_PLIES material_white=([+-][0-9.]+)")
PLIES_COUNT = re.compile(r"\bplies=(\d+)")


class TrainingController:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("チェスAI 連続強化学習")
        self.root.geometry("860x720")
        self.root.minsize(720, 600)
        self.process: subprocess.Popen[str] | None = None
        self.stop_requested = False
        state = self.load_state()
        self.completed_runs = int(state.get("completed_runs", 0))
        self.completed_epochs = int(state.get("completed_epochs", 0))
        self.white_wins = int(state.get("white_wins", 0))
        self.black_wins = int(state.get("black_wins", 0))
        self.draws = int(state.get("draws", 0))
        self.cutoffs = int(state.get("cutoffs", 0))
        self.fix_white_wins = int(state.get("fix_white_wins", 0))
        self.fix_black_wins = int(state.get("fix_black_wins", 0))
        self.fix_draws = int(state.get("fix_draws", 0))
        self.fix_cutoffs = int(state.get("fix_cutoffs", 0))
        self.resume_requested = bool(state.get("resume_requested", False))
        self.save_interval_var = tk.StringVar(value=str(state.get("save_interval", 100)))
        self.workers_var = tk.StringVar(value=str(state.get("selfplay_workers", 2)))
        self.status_var = tk.StringVar(value="停止中")
        self.counter_var = tk.StringVar()
        self.results_var = tk.StringVar()
        self.fix_results_var = tk.StringVar()
        self.saved_models_var = tk.StringVar(value="保存モデルを読み込み中…")
        self.update_counter()
        self.update_results()
        self.build_ui()
        self.load_log_history()
        self.refresh_saved_models()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        if self.resume_requested:
            self.status_var.set("前回の連続学習を自動再開します…")
            self.append_log("Windows再起動後の自動再開を準備中…")
            self.root.after(2000, self.start_training)

    def build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="AlphaZero 連続強化学習", font=("Yu Gothic UI", 16, "bold")).pack(pady=(0, 4))
        ttk.Label(frame, text="1回：並列自己対局・50シミュレーション・緊急上限300手・5エポック", foreground="#555555").pack(pady=(0, 4))
        ttk.Label(frame, textvariable=self.counter_var, font=("Yu Gothic UI", 10, "bold")).pack(pady=(0, 10))
        ttk.Label(frame, textvariable=self.results_var, font=("Yu Gothic UI", 10)).pack(pady=(0, 10))
        ttk.Label(frame, textvariable=self.fix_results_var, font=("Yu Gothic UI", 10, "bold")).pack(pady=(0, 10))

        settings = ttk.Frame(frame)
        settings.pack(fill="x", pady=(0, 9))
        ttk.Label(settings, text="世代モデルの保存間隔：").pack(side="left")
        self.interval_spinbox = ttk.Spinbox(settings, from_=1, to=1_000_000, textvariable=self.save_interval_var, width=9)
        self.interval_spinbox.pack(side="left")
        ttk.Label(settings, text="回ごと（例：100 / 500）").pack(side="left", padx=(5, 0))
        ttk.Label(settings, text="　自己対局の並列数：").pack(side="left")
        self.workers_combo = ttk.Combobox(settings, textvariable=self.workers_var, values=("1", "2", "3", "4"), state="readonly", width=4)
        self.workers_combo.pack(side="left")

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        self.start_button = ttk.Button(buttons, text="連続強化学習をスタート", command=self.start_training)
        self.start_button.pack(side="left", fill="x", expand=True, ipady=7)
        self.stop_button = ttk.Button(buttons, text="強化学習をストップ", command=self.stop_training, state="disabled")
        self.stop_button.pack(side="left", fill="x", expand=True, ipady=7, padx=(7, 0))
        ttk.Label(frame, textvariable=self.status_var, wraplength=810, justify="center").pack(fill="x", pady=(9, 10))

        lower = ttk.Panedwindow(frame, orient="horizontal")
        lower.pack(fill="both", expand=True)
        log_frame = ttk.Frame(lower)
        saved_frame = ttk.Frame(lower, padding=(10, 0, 0, 0))
        lower.add(log_frame, weight=3)
        lower.add(saved_frame, weight=2)

        ttk.Label(log_frame, text="実行ログ", font=("Yu Gothic UI", 10, "bold")).pack(anchor="w")
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, state="disabled", bg="#111820", fg="#d5e7d5", insertbackground="#ffffff", font=("Consolas", 9), relief="flat", padx=9, pady=8)
        self.log_text.pack(fill="both", expand=True, pady=(5, 0))

        ttk.Label(saved_frame, text="保存済み世代モデル", font=("Yu Gothic UI", 10, "bold")).pack(anchor="w")
        ttk.Label(saved_frame, textvariable=self.saved_models_var, foreground="#555555", wraplength=270).pack(anchor="w", pady=(2, 5))
        self.models_list = tk.Listbox(saved_frame, height=15, exportselection=False, font=("Consolas", 9))
        self.models_list.pack(fill="both", expand=True)
        ttk.Button(saved_frame, text="一覧を更新", command=self.refresh_saved_models).pack(fill="x", pady=(6, 3))
        ttk.Button(saved_frame, text="現在のモデルを手動保存", command=self.save_current_model).pack(fill="x", pady=3)
        ttk.Button(saved_frame, text="選択したモデルを削除", command=self.delete_selected_model).pack(fill="x", pady=3)
        ttk.Label(saved_frame, text="latest.msgpackは常に更新され、ここには表示されません。", foreground="#555555", wraplength=270).pack(anchor="w", pady=(6, 0))

    def load_state(self) -> dict[str, int]:
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def save_state(self, interval: int | None = None, workers: int | None = None) -> None:
        if interval is None:
            try:
                interval = max(1, int(self.save_interval_var.get()))
            except ValueError:
                interval = 100
        if workers is None:
            try:
                workers = int(self.workers_var.get())
            except ValueError:
                workers = 2
        payload = json.dumps(
            {
                "completed_runs": self.completed_runs,
                "completed_epochs": self.completed_epochs,
                "save_interval": interval,
                "selfplay_workers": workers,
                "resume_requested": self.resume_requested,
                "white_wins": self.white_wins,
                "black_wins": self.black_wins,
                "draws": self.draws,
                "cutoffs": self.cutoffs,
                "fix_white_wins": self.fix_white_wins,
                "fix_black_wins": self.fix_black_wins,
                "fix_draws": self.fix_draws,
                "fix_cutoffs": self.fix_cutoffs,
            },
            ensure_ascii=False,
            indent=2,
        )
        temporary = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(STATE_FILE)

    def update_counter(self) -> None:
        self.counter_var.set(f"完了回数：{self.completed_runs}回　累計：{self.completed_epochs}エポック")

    def update_results(self) -> None:
        total = self.white_wins + self.black_wins + self.draws + self.cutoffs
        self.results_var.set(
            f"自己対局成績（{total}局）：白勝ち {self.white_wins}　"
            f"黒勝ち {self.black_wins}　引き分け {self.draws}　"
            f"手数上限打ち切り {self.cutoffs}"
        )
        fix_total = (
            self.fix_white_wins
            + self.fix_black_wins
            + self.fix_draws
            + self.fix_cutoffs
        )
        self.fix_results_var.set(
            f"ループ対策後（{fix_total}局）：白勝ち {self.fix_white_wins}　"
            f"黒勝ち {self.fix_black_wins}　引き分け {self.fix_draws}　"
            f"上限打ち切り {self.fix_cutoffs}"
        )

    def record_game_result(self, result: str, message: str) -> None:
        cutoff = CUTOFF_SCORE.search(message)
        if cutoff:
            self.cutoffs += 1
            self.fix_cutoffs += 1
            score = float(cutoff.group(1))
            plies_match = PLIES_COUNT.search(message)
            plies = plies_match.group(1) if plies_match else "上限"
            label = f"{plies}手打ち切り（白視点の駒得評価 {score:+.2f}）"
        elif result == "1-0":
            self.white_wins += 1
            self.fix_white_wins += 1
            label = "白の勝ち"
        elif result == "0-1":
            self.black_wins += 1
            self.fix_black_wins += 1
            label = "黒の勝ち"
        else:
            self.draws += 1
            self.fix_draws += 1
            label = "引き分け"
        self.update_results()
        self.save_state()
        self.append_log(f"【対局結果】{label}（{result}）")

    def load_log_history(self) -> None:
        try:
            lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-500:]
        except FileNotFoundError:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", "\n".join(lines) + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def append_log(self, message: str) -> None:
        timestamped = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", timestamped + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")
        with LOG_FILE.open("a", encoding="utf-8") as log:
            log.write(timestamped + "\n")

    def wsl_process(self, command: str) -> subprocess.Popen[str]:
        return subprocess.Popen(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)

    def run_stage(self, label: str, command: str) -> None:
        if self.stop_requested:
            raise InterruptedError
        self.root.after(0, lambda: self.status_var.set(label))
        self.root.after(0, lambda: self.append_log(label))
        # 世代保存は「フォルダ作成 && コピー」の複合コマンドなので、
        # 先頭をexecで置換せず最後までシェルに実行させる。
        self.process = self.wsl_process(f"cd {PROJECT} && {command}")
        output: list[str] = []
        if self.process.stdout:
            for line in self.process.stdout:
                message = line.strip()
                if message:
                    output.append(message)
                    self.root.after(0, lambda message=message: self.append_log(message))
                    match = GAME_RESULT.match(message)
                    if match:
                        result = match.group(1)
                        self.root.after(
                            0,
                            lambda result=result, message=message: self.record_game_result(
                                result, message
                            ),
                        )
        exit_code = self.process.wait()
        self.process = None
        if self.stop_requested:
            raise InterruptedError
        if exit_code != 0:
            raise RuntimeError(output[-1] if output else f"終了コード {exit_code}")

    def start_training(self) -> None:
        if self.process is not None or self.stop_button["state"] == "normal":
            return
        try:
            interval = int(self.save_interval_var.get())
            if interval < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("保存間隔", "保存間隔には1以上の整数を入力してください。")
            return
        try:
            workers = int(self.workers_var.get())
            if workers not in (1, 2, 3, 4):
                raise ValueError
        except ValueError:
            messagebox.showwarning("並列数", "自己対局の並列数は1・2・3・4から選んでください。")
            return
        self.stop_requested = False
        self.resume_requested = True
        self.save_state(interval, workers)
        self.interval_spinbox.configure(state="disabled")
        self.workers_combo.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("連続学習を準備しています…")
        self.append_log(f"=== 連続強化学習を開始（自己対局{workers}並列・{interval}回ごとに世代保存） ===")
        threading.Thread(target=self.training_worker, args=(interval, workers), daemon=True).start()

    def training_worker(self, save_interval: int, workers: int) -> None:
        try:
            while not self.stop_requested:
                next_run = self.completed_runs + 1
                self.root.after(0, lambda next_run=next_run: self.append_log(f"--- 学習{next_run}回目を開始 ---"))
                self.run_stage(f"{next_run}回目：自己対局中（{workers}局並列・50シミュレーション・緊急上限300手・失敗時2回再試行）…", f"{PYTHON} -u parallel_selfplay.py --games {workers} --workers {workers} --simulations 50 --max-plies 300 --retries 2")
                self.run_stage(f"{next_run}回目：学習中（5エポック・直近200局）…", f"{PYTHON} -u train.py --epochs 5 --batch-size 32 --max-files 200")
                self.completed_runs = next_run
                self.completed_epochs += 5
                self.save_state(save_interval, workers)
                self.root.after(0, lambda run=next_run, epochs=self.completed_epochs: self.cycle_finished(run, epochs))
                if next_run % save_interval == 0:
                    filename = f"run-{next_run:06d}.msgpack"
                    self.run_stage(f"{next_run}回目の世代モデルを保存中…", f"mkdir -p {SAVED_MODELS} && cp checkpoints/latest.msgpack {SAVED_MODELS}/{filename}")
                    self.root.after(0, lambda filename=filename: self.snapshot_saved(filename))
        except InterruptedError:
            self.root.after(0, self.training_stopped)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.root.after(0, lambda message=message: self.training_failed(message))

    def cycle_finished(self, run: int, epochs: int) -> None:
        self.update_counter()
        self.status_var.set(f"{run}回完了。次の自己対局を開始します…")
        self.append_log(f"=== 学習{run}回目完了（累計{epochs}エポック） ===")

    def snapshot_saved(self, filename: str) -> None:
        self.append_log(f"世代モデル保存完了: {filename}")
        self.refresh_saved_models()

    def stop_training(self) -> None:
        if self.stop_button["state"] == "disabled":
            return
        self.stop_requested = True
        self.resume_requested = False
        self.save_state()
        self.stop_button.configure(state="disabled")
        self.status_var.set("停止しています…")
        threading.Thread(target=self.stop_worker, daemon=True).start()

    def stop_worker(self) -> None:
        kill_command = "pkill -TERM -f '^/home/user/chess/alphazero_chess/.venv/bin/python( -u)? (parallel_selfplay.py|selfplay.py|train.py)' 2>/dev/null || true"
        subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", kill_command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW, check=False)
        process = self.process
        if process is not None:
            try:
                process.terminate()
            except OSError:
                pass

    def reset_buttons(self) -> None:
        self.process = None
        self.interval_spinbox.configure(state="normal")
        self.workers_combo.configure(state="readonly")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def training_stopped(self) -> None:
        self.reset_buttons()
        self.status_var.set("停止しました。完了済みの学習結果は残ります。")
        self.append_log("=== 連続強化学習を停止 ===")

    def training_failed(self, message: str) -> None:
        self.reset_buttons()
        self.resume_requested = False
        self.save_state()
        self.status_var.set("エラーで停止しました。")
        self.append_log(f"エラー: {message}")
        messagebox.showerror("強化学習エラー", message)

    def refresh_saved_models(self) -> None:
        self.saved_models_var.set("保存モデルを読み込み中…")
        threading.Thread(target=self.saved_models_worker, daemon=True).start()

    def saved_models_worker(self) -> None:
        command = f"mkdir -p {SAVED_MODELS}; find {SAVED_MODELS} -maxdepth 1 -type f -name 'run-*.msgpack' -printf '%f|%s\\n' | sort -r"
        result = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command], capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW, check=False)
        if result.returncode != 0:
            self.root.after(0, lambda: self.saved_models_var.set("一覧の取得に失敗しました"))
            return
        items: list[tuple[str, int]] = []
        for line in result.stdout.splitlines():
            try:
                name, size = line.rsplit("|", 1)
                if MODEL_NAME.fullmatch(name):
                    items.append((name, int(size)))
            except ValueError:
                continue
        self.root.after(0, lambda items=items: self.show_saved_models(items))

    def show_saved_models(self, items: list[tuple[str, int]]) -> None:
        self.models_list.delete(0, "end")
        for name, size in items:
            self.models_list.insert("end", f"{name}  ({size / 1024 / 1024:.1f} MB)")
        self.saved_models_var.set(f"{len(items)}個・合計 {sum(size for _, size in items) / 1024 / 1024:.1f} MB")

    def save_current_model(self) -> None:
        if self.completed_runs < 1:
            messagebox.showinfo("保存", "まだ完了した学習モデルがありません。")
            return
        filename = f"run-{self.completed_runs:06d}.msgpack"
        if not messagebox.askyesno("現在のモデルを保存", f"現在のモデルを {filename} として保存しますか？\n同じ番号がある場合は上書きします。"):
            return
        threading.Thread(target=self.save_current_worker, args=(filename,), daemon=True).start()

    def save_current_worker(self, filename: str) -> None:
        command = f"mkdir -p {SAVED_MODELS} && cp {PROJECT}/checkpoints/latest.msgpack {SAVED_MODELS}/{filename}"
        result = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW, check=False)
        if result.returncode == 0:
            self.root.after(0, lambda: self.snapshot_saved(filename))
        else:
            self.root.after(0, lambda: messagebox.showerror("保存エラー", "モデルを保存できませんでした。"))

    def delete_selected_model(self) -> None:
        selection = self.models_list.curselection()
        if not selection:
            messagebox.showinfo("削除", "削除する世代モデルを選択してください。")
            return
        filename = self.models_list.get(selection[0]).split("  (", 1)[0]
        if not MODEL_NAME.fullmatch(filename):
            messagebox.showerror("削除エラー", "削除対象の名前が不正です。")
            return
        if not messagebox.askyesno("世代モデルを削除", f"{filename} を完全に削除しますか？\nこの操作は元に戻せません。"):
            return
        threading.Thread(target=self.delete_model_worker, args=(filename,), daemon=True).start()

    def delete_model_worker(self, filename: str) -> None:
        result = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "rm", "--", f"{SAVED_MODELS}/{filename}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW, check=False)
        if result.returncode == 0:
            self.root.after(0, lambda: self.append_log(f"世代モデルを削除: {filename}"))
            self.root.after(0, self.refresh_saved_models)
        else:
            self.root.after(0, lambda: messagebox.showerror("削除エラー", "モデルを削除できませんでした。"))

    def on_close(self) -> None:
        if self.stop_button["state"] == "normal":
            if not messagebox.askyesno("学習を停止しますか？", "連続強化学習が動いています。停止して画面を閉じますか？"):
                return
            self.stop_requested = True
            self.resume_requested = False
            self.save_state()
            self.stop_worker()
        self.root.destroy()


def main() -> None:
    if "--resume-only" in sys.argv:
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if not state.get("resume_requested", False):
            return
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    TrainingController(root)
    root.mainloop()


if __name__ == "__main__":
    main()
