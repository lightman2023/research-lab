"""GUI for v2 self-play trials with an embedded terminal-style log."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import subprocess
import threading
import tkinter as tk
import uuid
from tkinter import messagebox, scrolledtext, ttk

PROJECT = "/home/user/chess/alphazero_chess_v2"
PYTHON = f"{PROJECT}/.venv/bin/python"
SAVED = f"{PROJECT}/checkpoints/saved"
MODEL_NAME = re.compile(r"^(?:latest|cycle-\d{6,})\.msgpack$")
RESULT = re.compile(
    r"^Game \d+: (\S+) termination=([A-Z_]+) plies=(\d+) -> (.+)$"
)


class TrialApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.process: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self.close_when_stopped = False
        self.output_dir = ""
        self.script_name = ""
        root.title("v2 ループ対策 試験対局")
        root.geometry("800x620")
        root.minsize(700, 520)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        frame = ttk.Frame(root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame, text="v2 ループ対策 試験対局",
            font=("Yu Gothic UI", 16, "bold"),
        ).pack(anchor="w", pady=(0, 5))
        ttk.Label(
            frame,
            text="同じモデルで従来方式と反復対策方式の自己対局を試します。試験結果は通常の学習に加えません。",
            wraplength=760,
        ).pack(anchor="w", pady=(0, 12))

        self.mode = tk.StringVar(value="反復対策")
        self.model = tk.StringVar(value="cycle-000119.msgpack")
        self.games = tk.StringVar(value="4")
        self.simulations = tk.StringVar(value="50")
        self.max_plies = tk.StringVar(value="300")
        self.status = tk.StringVar(value="準備完了")
        self.score = tk.StringVar(value="結果：未実行")

        settings = ttk.LabelFrame(frame, text="試験条件", padding=10)
        settings.pack(fill="x")
        row = ttk.Frame(settings)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="方式", width=14).pack(side="left")
        self.mode_box = ttk.Combobox(
            row, textvariable=self.mode, state="readonly",
            values=("反復対策", "従来方式"), width=18,
        )
        self.mode_box.pack(side="left")
        ttk.Label(row, text="　モデル").pack(side="left")
        self.model_box = ttk.Combobox(
            row, textvariable=self.model, state="readonly", width=25
        )
        self.model_box.pack(side="left", padx=(4, 0))
        self.refresh_button = ttk.Button(
            row, text="一覧を更新", command=self.refresh_models
        )
        self.refresh_button.pack(side="left", padx=(8, 0))
        row = ttk.Frame(settings)
        row.pack(fill="x", pady=5)
        self.entries = []
        for label, variable, width in (
            ("対局数", self.games, 7),
            ("1手の探索回数", self.simulations, 8),
            ("最大手数", self.max_plies, 8),
        ):
            ttk.Label(row, text=label).pack(side="left", padx=(0, 4))
            entry = ttk.Entry(row, textvariable=variable, width=width)
            entry.pack(side="left", padx=(0, 18))
            self.entries.append(entry)

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(10, 0))
        self.start_button = ttk.Button(
            buttons, text="試験対局を実行", command=self.start_trial
        )
        self.start_button.pack(side="left", fill="x", expand=True, ipady=6)
        self.stop_button = ttk.Button(
            buttons, text="ストップ", command=self.stop_trial, state="disabled"
        )
        self.stop_button.pack(
            side="left", fill="x", expand=True, padx=(7, 0), ipady=6
        )

        ttk.Label(frame, textvariable=self.status).pack(
            anchor="w", pady=(10, 3)
        )
        ttk.Label(
            frame, textvariable=self.score,
            font=("Yu Gothic UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(frame, text="対局ログ", font=("Yu Gothic UI", 10, "bold")).pack(
            anchor="w"
        )
        self.log = scrolledtext.ScrolledText(
            frame, state="disabled", bg="#111820", fg="#d5e7d5",
            insertbackground="#ffffff", font=("Consolas", 9),
            relief="flat", padx=9, pady=8,
        )
        self.log.pack(fill="both", expand=True, pady=(5, 0))
        self.refresh_models()

    def append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"[{datetime.now():%H:%M:%S}] {message}\n")
        self.log.configure(state="disabled")
        self.log.see("end")

    def refresh_models(self) -> None:
        try:
            result = subprocess.run(
                [
                    "wsl.exe", "-d", "Ubuntu", "--", "find", SAVED,
                    "-maxdepth", "1", "-type", "f",
                    "-name", "cycle-*.msgpack",
                ],
                capture_output=True, text=True, timeout=20, check=True,
            )
            names = sorted(
                Path(line.strip()).name for line in result.stdout.splitlines()
                if MODEL_NAME.fullmatch(Path(line.strip()).name)
            )
            names.append("latest.msgpack")
            self.model_box["values"] = names
            if self.model.get() not in names:
                self.model.set("latest.msgpack")
        except (OSError, subprocess.SubprocessError) as error:
            messagebox.showerror("モデル一覧", str(error))

    def set_running(self, running: bool) -> None:
        self.mode_box.configure(state="disabled" if running else "readonly")
        self.model_box.configure(state="disabled" if running else "readonly")
        self.refresh_button.configure(state="disabled" if running else "normal")
        for entry in self.entries:
            entry.configure(state="disabled" if running else "normal")
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def start_trial(self) -> None:
        if not MODEL_NAME.fullmatch(self.model.get()):
            messagebox.showwarning("モデル", "モデルを選んでください。")
            return
        try:
            games = int(self.games.get())
            simulations = int(self.simulations.get())
            max_plies = int(self.max_plies.get())
            if not (1 <= games <= 100 and 1 <= simulations <= 1000
                    and 20 <= max_plies <= 512):
                raise ValueError
        except ValueError:
            messagebox.showwarning(
                "設定値",
                "対局数1～100、探索回数1～1000、最大手数20～512を入力してください。",
            )
            return
        probe = subprocess.run(
            [
                "wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc",
                "pgrep -f '[p]arallel_selfplay.py|[/]selfplay.py|[/]selfplay_loopfix.py|[/]train.py|[/]arena.py' >/dev/null",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if probe.returncode == 0:
            messagebox.showwarning(
                "対局・学習が実行中です",
                "先に実行中の対局または学習を停止してください。",
            )
            return

        mode = "loopfix" if self.mode.get() == "反復対策" else "baseline"
        self.script_name = (
            "selfplay_loopfix.py" if mode == "loopfix" else "selfplay.py"
        )
        model_path = (
            f"{PROJECT}/checkpoints/latest.msgpack"
            if self.model.get() == "latest.msgpack"
            else f"{SAVED}/{self.model.get()}"
        )
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.output_dir = f"{PROJECT}/arena_results/{mode}-trial-{run_id}"
        command = [
            "wsl.exe", "-d", "Ubuntu", "--", PYTHON, "-u",
            f"{PROJECT}/{self.script_name}",
            "--games", str(games),
            "--simulations", str(simulations),
            "--max-plies", str(max_plies),
            "--checkpoint", model_path,
            "--output-dir", self.output_dir,
        ]
        self.stop_requested = False
        self.set_running(True)
        self.status.set(f"{self.mode.get()}：{games}局を実行中")
        self.score.set("結果：対局中")
        self.append_log(
            f"=== 開始：{self.mode.get()} / {self.model.get()} / "
            f"{games}局 / 1手{simulations}探索 / 最大{max_plies}手 ==="
        )
        self.append_log(f"保存先：{self.output_dir}")
        threading.Thread(
            target=self.run_worker, args=(command, games), daemon=True
        ).start()

    def run_worker(self, command: list[str], games: int) -> None:
        outcomes: Counter[str] = Counter()
        terminations: Counter[str] = Counter()
        plies: list[int] = []
        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            if self.stop_requested:
                self.stop_worker()
            assert self.process.stdout is not None
            for line in self.process.stdout:
                line = line.rstrip()
                if line:
                    self.root.after(0, lambda text=line: self.append_log(text))
                match = RESULT.match(line)
                if match:
                    outcomes[match.group(1)] += 1
                    terminations[match.group(2)] += 1
                    plies.append(int(match.group(3)))
            exit_code = self.process.wait()
            self.process = None
            self.root.after(
                0, lambda: self.finish_trial(
                    games, outcomes, terminations, plies, exit_code
                )
            )
        except Exception as error:
            self.process = None
            self.root.after(
                0, lambda err=error: self.fail_trial(err)
            )

    def finish_trial(
        self, games: int, outcomes: Counter[str],
        terminations: Counter[str], plies: list[int], exit_code: int,
    ) -> None:
        self.set_running(False)
        if self.stop_requested:
            self.status.set("試験対局を停止しました。完了済みの結果は残ります。")
        elif exit_code == 0 and len(plies) == games:
            self.status.set("試験対局が完了しました。")
        else:
            self.status.set(f"試験対局は途中で終了しました（終了コード {exit_code}）。")
        self.score.set(
            f"完了 {len(plies)}/{games}局：白 {outcomes['1-0']}勝 / "
            f"黒 {outcomes['0-1']}勝 / 引き分け {outcomes['1/2-1/2']}"
        )
        self.append_log(f"終局理由：{dict(terminations)}")
        if plies:
            self.append_log(f"平均手数：{sum(plies) / len(plies):.1f}")
        self.append_log(f"=== 終了：結果保存先 {self.output_dir} ===")
        if self.close_when_stopped:
            self.root.destroy()

    def fail_trial(self, error: Exception) -> None:
        self.set_running(False)
        self.status.set("起動に失敗しました。")
        self.append_log(f"エラー：{type(error).__name__}: {error}")
        if self.close_when_stopped:
            self.root.destroy()
        else:
            messagebox.showerror("試験対局エラー", str(error))

    def stop_trial(self) -> None:
        self.stop_requested = True
        self.stop_button.configure(state="disabled")
        self.status.set("試験対局を停止しています…")
        threading.Thread(target=self.stop_worker, daemon=True).start()

    def stop_worker(self) -> None:
        if self.output_dir and self.script_name:
            pattern = (
                f"[/]{self.script_name}.*--output-dir {self.output_dir}"
            )
            subprocess.run(
                [
                    "wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc",
                    f"pkill -TERM -f '{pattern}' 2>/dev/null || true",
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False,
            )
        if self.process is not None:
            try:
                self.process.terminate()
            except OSError:
                pass

    def on_close(self) -> None:
        if self.start_button["state"] == "disabled":
            if not messagebox.askyesno(
                "試験対局を停止しますか？",
                "実行中の試験対局を停止して閉じますか？",
            ):
                return
            self.close_when_stopped = True
            self.stop_trial()
            return
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    TrialApp(root)
    root.mainloop()

