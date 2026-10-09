"""Play against checkpoints from the separate scaled AlphaZero v2 experiment."""

from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import chess
import chess.engine


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
ENGINE_DIR = APP_DIR / "engine"
SETTINGS_FILE = APP_DIR / "settings.json"
LOG_FILE = APP_DIR / "chess_app.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)
LOGGER = logging.getLogger("chess_app")
POLICY_LINE = re.compile(
    r"^([a-h][1-8][a-h][1-8][qrbn]?)\s+.*\(P:\s*([0-9.]+)%\)"
)

LIGHT = "#f0d9b5"
DARK = "#b58863"
SELECTED = "#f6f669"
LAST_MOVE = "#cdd26a"
LEGAL_MOVE = "#79a65a"
CAPTURE_TARGET = "#c8463a"
PIECES = {
    chess.PAWN: ("♙", "♟"),
    chess.KNIGHT: ("♘", "♞"),
    chess.BISHOP: ("♗", "♝"),
    chess.ROOK: ("♖", "♜"),
    chess.QUEEN: ("♕", "♛"),
    chess.KING: ("♔", "♚"),
}


class ChessApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("縮小AlphaZero v2 対戦")
        self.root.minsize(820, 650)

        MODELS_DIR.mkdir(exist_ok=True)
        ENGINE_DIR.mkdir(exist_ok=True)
        self.settings = self.load_settings()
        self.board = chess.Board()
        self.engine: chess.engine.SimpleEngine | None = None
        self.az_process: subprocess.Popen | None = None
        self.engine_log = None
        self.engine_busy = False
        self.animating = False
        self.animation: dict[str, object] | None = None
        self.result_overlay: tk.Frame | None = None
        self.result_shown = False
        self.selected: chess.Square | None = None
        self.human_color = chess.WHITE
        self.flipped = False

        self.model_var = tk.StringVar()
        self.color_var = tk.StringVar(value=self.settings.get("human_color", "白"))
        self.status_var = tk.StringVar(value="モデルを選択して「新しい対局」を押してください。")
        self.history_var = tk.StringVar()
        self.build_ui()
        self.refresh_models(self.settings.get("model", ""))
        self.draw_board()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        board_frame = ttk.Frame(outer)
        board_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(board_frame, width=640, height=640, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_board_click)
        self.canvas.bind("<Configure>", lambda _event: self.draw_board())

        panel = ttk.Frame(outer, width=300, padding=(14, 0, 0, 0))
        panel.pack(side="right", fill="y")

        ttk.Label(panel, text="対戦モデル", font=("Yu Gothic UI", 12, "bold")).pack(anchor="w")
        self.model_combo = ttk.Combobox(panel, textvariable=self.model_var, state="readonly", width=35)
        self.model_combo.pack(fill="x", pady=(6, 4))
        ttk.Button(panel, text="モデル一覧を更新", command=self.refresh_models).pack(fill="x")
        ttk.Button(panel, text="別のモデルを選ぶ…", command=self.choose_model).pack(fill="x", pady=(4, 14))

        ttk.Label(panel, text="あなたの色", font=("Yu Gothic UI", 11, "bold")).pack(anchor="w")
        ttk.Combobox(
            panel,
            textvariable=self.color_var,
            state="readonly",
            values=("白", "黒", "ランダム"),
        ).pack(fill="x", pady=(6, 12))

        ttk.Button(panel, text="新しい対局", command=self.new_game).pack(fill="x", ipady=6)
        ttk.Button(panel, text="投了", command=self.resign).pack(fill="x", pady=(5, 16))

        ttk.Separator(panel).pack(fill="x", pady=5)
        ttk.Label(panel, text="状態", font=("Yu Gothic UI", 11, "bold")).pack(anchor="w", pady=(8, 3))
        ttk.Label(panel, textvariable=self.status_var, wraplength=280).pack(anchor="w", fill="x")

        ttk.Label(panel, text="指し手", font=("Yu Gothic UI", 11, "bold")).pack(anchor="w", pady=(16, 3))
        ttk.Label(panel, textvariable=self.history_var, wraplength=280, justify="left").pack(anchor="w")

        ttk.Label(
            panel,
            text="今後の学習済みモデルは models フォルダへ入れ、一覧を更新してください。",
            wraplength=280,
            foreground="#555555",
        ).pack(side="bottom", anchor="w", pady=(20, 0))

    def load_settings(self) -> dict[str, str]:
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_settings(self) -> None:
        data = dict(self.settings)
        data.update({"model": self.model_var.get(), "human_color": self.color_var.get()})
        SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def available_models(self) -> list[Path]:
        models = list(MODELS_DIR.glob("*.pb.gz")) + list(MODELS_DIR.glob("*.msgpack"))
        return sorted(models, key=lambda path: path.name.lower())

    def refresh_models(self, preferred: str | None = None) -> None:
        models = self.available_models()
        names = [model.name for model in models]
        self.model_combo["values"] = names
        wanted = preferred if preferred is not None else self.model_var.get()
        if wanted in names:
            self.model_var.set(wanted)
        elif names:
            self.model_var.set(names[0])
        else:
            self.model_var.set("")

    def choose_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="対戦モデルを選択",
            initialdir=MODELS_DIR,
            filetypes=(
                ("対応モデル", "*.pb.gz *.msgpack"),
                ("Lc0 network", "*.pb.gz"),
                ("AlphaZero checkpoint", "*.msgpack"),
                ("すべてのファイル", "*.*"),
            ),
        )
        if not selected:
            return
        source = Path(selected)
        destination = MODELS_DIR / source.name
        if source.resolve() != destination.resolve():
            destination.write_bytes(source.read_bytes())
        self.refresh_models(destination.name)

    def engine_path(self) -> Path:
        configured = self.settings.get("engine", "")
        if configured and Path(configured).is_file():
            return Path(configured)
        candidates = list(ENGINE_DIR.rglob("lc0.exe"))
        if candidates:
            return candidates[0]
        selected = filedialog.askopenfilename(
            title="lc0.exeを選択",
            filetypes=(("Lc0 engine", "lc0.exe"), ("実行ファイル", "*.exe")),
        )
        if not selected:
            raise FileNotFoundError("lc0.exeが選択されていません。")
        self.settings["engine"] = selected
        return Path(selected)

    def stop_engine(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None
        if self.az_process is not None:
            try:
                if self.az_process.stdin:
                    self.az_process.stdin.write(json.dumps({"command": "quit"}) + "\n")
                    self.az_process.stdin.flush()
                self.az_process.wait(timeout=3)
            except Exception:
                self.az_process.kill()
            self.az_process = None
        if self.engine_log is not None:
            self.engine_log.close()
            self.engine_log = None

    @staticmethod
    def windows_to_wsl(path: Path) -> str:
        resolved = path.resolve().as_posix()
        if len(resolved) >= 3 and resolved[1:3] == ":/":
            return f"/mnt/{resolved[0].lower()}/{resolved[3:]}"
        raise ValueError(f"WSLへ渡せないモデルパスです: {path}")

    def start_alphazero_engine(self, model: Path) -> None:
        linux_model = self.windows_to_wsl(model)
        server = "/home/user/chess/alphazero_chess_v2/serve.py"
        python = "/home/user/chess/alphazero_chess_v2/.venv/bin/python"
        self.engine_log = (APP_DIR / "alphazero.log").open("a", encoding="utf-8")
        command = [
            "wsl.exe", "-d", "Ubuntu", "--", python, server,
            "--checkpoint", linux_model, "--simulations", "50",
        ]
        LOGGER.info("Starting AlphaZero bridge model=%s", model)
        self.az_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.engine_log,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        response = self.az_process.stdout.readline() if self.az_process.stdout else ""
        if not response:
            raise RuntimeError("AlphaZeroエンジンを起動できませんでした。alphazero.logを確認してください。")
        payload = json.loads(response)
        if payload.get("status") != "ready":
            raise RuntimeError(payload.get("error", "AlphaZeroエンジンの準備に失敗しました。"))

    def start_engine(self, model: Path) -> None:
        self.stop_engine()
        if model.suffix.lower() == ".msgpack":
            self.start_alphazero_engine(model)
            return
        path = self.engine_path()
        LOGGER.info("Starting engine=%s model=%s", path, model)
        self.engine_log = (APP_DIR / "lc0.log").open("a", encoding="utf-8")
        # 起動時間は制限せず、CUDAの初期化完了を待つ。
        self.engine = chess.engine.SimpleEngine.popen_uci(
            [str(path), f"--weights={model}"],
            timeout=None,
            stderr=self.engine_log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        options: dict[str, int | float | bool] = {}
        if "VerboseMoveStats" in self.engine.options:
            options["VerboseMoveStats"] = True
        if "PolicyTemperature" in self.engine.options:
            # 学習済みの方策分布をそのまま使う。
            options["PolicyTemperature"] = 1.0
        if "Temperature" in self.engine.options:
            options["Temperature"] = 1.0
        if "TempDecayMoves" in self.engine.options:
            options["TempDecayMoves"] = 0
        if "TempCutoffMove" in self.engine.options:
            options["TempCutoffMove"] = 1000
        if options:
            self.engine.configure(options)

    def new_game(self) -> None:
        if self.engine_busy or self.animating:
            return
        if not self.model_var.get():
            messagebox.showwarning(
                "モデルがありません", "modelsフォルダへ.pb.gzまたは.msgpackモデルを入れてください。"
            )
            return
        model = MODELS_DIR / self.model_var.get()
        self.close_result_overlay()
        self.result_shown = False
        try:
            self.start_engine(model)
        except Exception as error:
            messagebox.showerror("エンジン起動エラー", str(error))
            return

        choice = self.color_var.get()
        if choice == "ランダム":
            self.human_color = random.choice((chess.WHITE, chess.BLACK))
        else:
            self.human_color = chess.WHITE if choice == "白" else chess.BLACK
        self.flipped = self.human_color == chess.BLACK
        self.board.reset()
        self.selected = None
        self.history_var.set("")
        self.status_var.set(f"あなたは{'白' if self.human_color else '黒'}です。")
        self.save_settings()
        self.draw_board()
        if self.board.turn != self.human_color:
            self.request_ai_move()

    def on_board_click(self, event: tk.Event) -> None:
        if self.engine_busy or self.animating or self.board.is_game_over() or self.board.turn != self.human_color:
            return
        size = min(self.canvas.winfo_width(), self.canvas.winfo_height()) / 8
        if size <= 0:
            return
        file_index = int(event.x // size)
        rank_index = 7 - int(event.y // size)
        if self.flipped:
            file_index = 7 - file_index
            rank_index = 7 - rank_index
        if not (0 <= file_index < 8 and 0 <= rank_index < 8):
            return
        square = chess.square(file_index, rank_index)
        piece = self.board.piece_at(square)

        if self.selected is None:
            if piece and piece.color == self.human_color:
                self.selected = square
        elif square == self.selected:
            self.selected = None
        elif piece and piece.color == self.human_color:
            self.selected = square
        else:
            move = chess.Move(self.selected, square)
            if self.board.piece_type_at(self.selected) == chess.PAWN and chess.square_rank(square) in (0, 7):
                promotion_moves = [
                    chess.Move(self.selected, square, promotion=piece_type)
                    for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
                ]
                legal_promotions = [move for move in promotion_moves if move in self.board.legal_moves]
                if legal_promotions:
                    chosen = self.choose_promotion(legal_promotions)
                    if chosen is None:
                        self.status_var.set("昇格をキャンセルしました。")
                        self.selected = None
                        self.draw_board()
                        return
                    move = chosen
            if move in self.board.legal_moves:
                self.selected = None
                self.animate_move(move, self.after_human_animation)
                return
            self.status_var.set("その手は指せません。")
            self.selected = None
        self.draw_board()

    def choose_promotion(self, legal_moves: list[chess.Move]) -> chess.Move | None:
        """ポーン昇格先をクイーン・ルーク・ビショップ・ナイトから選ぶ。"""
        selected_move: list[chess.Move | None] = [None]
        dialog = tk.Toplevel(self.root)
        dialog.title("ポーンの昇格")
        dialog.configure(bg="#17212b")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog,
            text="昇格する駒を選んでください",
            font=("Yu Gothic UI", 15, "bold"),
            fg="#ffffff",
            bg="#17212b",
        ).pack(padx=25, pady=(22, 14))
        buttons = tk.Frame(dialog, bg="#17212b")
        buttons.pack(padx=20, pady=(0, 22))
        names = {
            chess.QUEEN: "クイーン",
            chess.ROOK: "ルーク",
            chess.BISHOP: "ビショップ",
            chess.KNIGHT: "ナイト",
        }

        def select(move: chess.Move) -> None:
            selected_move[0] = move
            dialog.destroy()

        for move in legal_moves:
            symbol = PIECES[move.promotion][1]
            button = tk.Button(
                buttons,
                text=f"{symbol}\n{names[move.promotion]}",
                command=lambda move=move: select(move),
                font=("Segoe UI Symbol", 19, "bold"),
                fg="#ffffff",
                bg="#34495e",
                activeforeground="#ffffff",
                activebackground="#49647d",
                width=8,
                height=3,
                relief="flat",
                cursor="hand2",
            )
            button.pack(side="left", padx=5)

        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        self.root.wait_window(dialog)
        return selected_move[0]

    def request_ai_move(self) -> None:
        if self.engine is None and self.az_process is None:
            return
        self.engine_busy = True
        self.status_var.set(f"{self.model_var.get()} が考えています…")
        self.draw_board()
        board_copy = self.board.copy()

        def worker() -> None:
            try:
                LOGGER.info("AI search start fen=%s", board_copy.fen())
                if self.az_process is not None:
                    move = self.request_alphazero_move(board_copy)
                    LOGGER.info("AlphaZero MCTS move=%s", move)
                else:
                    # Dala公式と同じく、深さ1の方策確率から重み付き抽選する。
                    move = self.sample_policy_move(board_copy)
                    LOGGER.info("Dala policy sample move=%s", move)
                self.root.after(0, lambda move=move: self.finish_ai_move(move))
            except Exception as error:
                # Pythonはexcept終了時にerror変数を消すため、既定引数へ値を保持する。
                message = f"{type(error).__name__}: {error}"
                LOGGER.exception("AI search failed")
                self.root.after(0, lambda message=message: self.engine_failed(message))

        threading.Thread(target=worker, daemon=True).start()

    def request_alphazero_move(self, board: chess.Board) -> chess.Move:
        if self.az_process is None or self.az_process.stdin is None or self.az_process.stdout is None:
            raise RuntimeError("AlphaZeroエンジンが起動していません。")
        request = {
            "fen": board.fen(),
            "moves": [move.uci() for move in board.move_stack],
            "temperature": 0.25,
        }
        self.az_process.stdin.write(json.dumps(request) + "\n")
        self.az_process.stdin.flush()
        response = self.az_process.stdout.readline()
        if not response:
            raise RuntimeError("AlphaZeroエンジンが応答せず終了しました。")
        payload = json.loads(response)
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return chess.Move.from_uci(payload["move"])

    def sample_policy_move(self, board: chess.Board) -> chess.Move:
        """Lc0が出力する全合法手のP値を読み、確率に比例して1手を選ぶ。"""
        if self.engine is None:
            raise RuntimeError("AIエンジンが起動していません。")
        moves: list[chess.Move] = []
        weights: list[float] = []
        analysis = self.engine.analysis(
            board, chess.engine.Limit(nodes=1), info=chess.engine.INFO_ALL
        )
        for info in analysis:
            line = info.get("string")
            if not line:
                continue
            match = POLICY_LINE.match(line)
            if not match:
                continue
            try:
                move = chess.Move.from_uci(match.group(1))
                probability = float(match.group(2))
            except ValueError:
                continue
            if move in board.legal_moves and probability > 0:
                moves.append(move)
                weights.append(probability)

        if not moves:
            LOGGER.warning("Policy values unavailable; falling back to engine best move")
            return self.engine.play(board, chess.engine.Limit(nodes=1)).move
        LOGGER.info(
            "Policy candidates=%s",
            ", ".join(f"{move.uci()}:{weight:.2f}%" for move, weight in zip(moves, weights)),
        )
        return random.choices(moves, weights=weights, k=1)[0]

    def finish_ai_move(self, move: chess.Move) -> None:
        if move in self.board.legal_moves:
            self.animate_move(move, self.after_ai_animation)
        else:
            self.engine_busy = False
            self.status_var.set("AIから不正な指し手が返されました。")

    def after_human_animation(self) -> None:
        self.after_move()
        if not self.board.is_game_over():
            self.request_ai_move()

    def after_ai_animation(self) -> None:
        self.engine_busy = False
        self.after_move()

    def square_center(self, square: chess.Square) -> tuple[float, float, float]:
        size = min(self.canvas.winfo_width(), self.canvas.winfo_height()) / 8
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        screen_file = 7 - file_index if self.flipped else file_index
        screen_rank = rank_index if self.flipped else 7 - rank_index
        return (screen_file + 0.5) * size, (screen_rank + 0.5) * size, size

    def piece_appearance(self, piece: chess.Piece) -> tuple[str, str, bool]:
        """双方とも塗りつぶし駒を使い、自分側だけ白＋黒縁にする。"""
        symbol = PIECES[piece.piece_type][1]
        is_human = piece.color == self.human_color
        return symbol, "#ffffff" if is_human else "#171717", is_human

    def draw_piece_text(
        self, x: float, y: float, size: float, symbol: str, fill: str, outlined: bool
    ) -> None:
        font = ("Segoe UI Symbol", max(20, int(size * 0.67)))
        if outlined:
            # 白駒が明るいマスでも見えるよう、8方向へ黒い縁を描く。
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, -1), (-1, 1), (1, 1)):
                self.canvas.create_text(x + dx, y + dy, text=symbol, font=font, fill="#222222")
        self.canvas.create_text(x, y, text=symbol, font=font, fill=fill)

    def animate_move(self, move: chess.Move, on_complete) -> None:
        """盤面を更新し、移動する駒だけを滑らかにフレーム補間する。"""
        piece = self.board.piece_at(move.from_square)
        if piece is None:
            return
        symbol, piece_fill, outlined = self.piece_appearance(piece)
        self.board.push(move)
        self.animating = True
        self.animation = {
            "to_square": move.to_square,
            "symbol": symbol,
            "fill": piece_fill,
            "outlined": outlined,
            "from": self.square_center(move.from_square),
            "to": self.square_center(move.to_square),
        }
        total_frames = 22
        frame_delay_ms = 15

        def render_frame(frame: int) -> None:
            if self.animation is None:
                return
            # smoothstep補間: 始点と終点で速度が自然に0になる。
            progress = min(1.0, frame / total_frames)
            eased = progress * progress * (3.0 - 2.0 * progress)
            from_x, from_y, from_size = self.animation["from"]
            to_x, to_y, to_size = self.animation["to"]
            x = from_x + (to_x - from_x) * eased
            y = from_y + (to_y - from_y) * eased
            size = from_size + (to_size - from_size) * eased
            self.draw_board()
            # 小さな影を付け、移動中の駒を盤面からわずかに浮かせる。
            self.canvas.create_text(
                x + 2,
                y + 3,
                text=self.animation["symbol"],
                font=("Segoe UI Symbol", max(20, int(size * 0.67))),
                fill="#5a4638",
                tags="moving_piece",
            )
            self.draw_piece_text(
                x,
                y,
                size,
                self.animation["symbol"],
                self.animation["fill"],
                self.animation["outlined"],
            )
            if frame < total_frames:
                self.root.after(frame_delay_ms, render_frame, frame + 1)
            else:
                self.animation = None
                self.animating = False
                self.draw_board()
                on_complete()

        render_frame(0)

    def engine_failed(self, error: str) -> None:
        self.engine_busy = False
        self.status_var.set("AIエラー")
        messagebox.showerror("AIエラー", error)

    def after_move(self) -> None:
        self.update_history()
        self.draw_board()
        if self.board.is_game_over():
            outcome = self.board.outcome()
            self.status_var.set(f"対局終了: {self.board.result()} ({outcome.termination.name})")
            self.show_result_overlay(outcome)
        elif self.board.turn == self.human_color:
            self.status_var.set("あなたの番です。駒をクリックしてください。")

    def update_history(self) -> None:
        replay = chess.Board()
        entries: list[str] = []
        for index, move in enumerate(self.board.move_stack):
            san = replay.san(move)
            if index % 2 == 0:
                entries.append(f"{index // 2 + 1}. {san}")
            else:
                entries[-1] += f" {san}"
            replay.push(move)
        self.history_var.set("  ".join(entries[-12:]))

    def draw_board(self) -> None:
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        size = min(width, height) / 8
        if size <= 1:
            return
        last = self.board.peek() if self.board.move_stack else None
        legal_targets = {move.to_square for move in self.board.legal_moves if move.from_square == self.selected}

        for screen_rank in range(8):
            for screen_file in range(8):
                file_index = 7 - screen_file if self.flipped else screen_file
                rank_index = screen_rank if self.flipped else 7 - screen_rank
                square = chess.square(file_index, rank_index)
                color = LIGHT if (file_index + rank_index) % 2 else DARK
                if last and square in (last.from_square, last.to_square):
                    color = LAST_MOVE
                if square == self.selected:
                    color = SELECTED
                x1, y1 = screen_file * size, screen_rank * size
                self.canvas.create_rectangle(x1, y1, x1 + size, y1 + size, fill=color, outline=color)
                piece = self.board.piece_at(square)
                moving_to = self.animation.get("to_square") if self.animation else None
                if piece and square != moving_to:
                    symbol, piece_fill, outlined = self.piece_appearance(piece)
                    self.draw_piece_text(
                        x1 + size / 2,
                        y1 + size / 2,
                        size,
                        symbol,
                        piece_fill,
                        outlined,
                    )
                    if square in legal_targets:
                        inset = size * 0.08
                        self.canvas.create_oval(
                            x1 + inset,
                            y1 + inset,
                            x1 + size - inset,
                            y1 + size - inset,
                            outline=CAPTURE_TARGET,
                            width=max(3, int(size * 0.055)),
                        )
                        self.canvas.create_rectangle(
                            x1 + 2,
                            y1 + 2,
                            x1 + size - 2,
                            y1 + size - 2,
                            outline=CAPTURE_TARGET,
                            width=2,
                        )
                elif square in legal_targets:
                    radius = size * 0.12
                    self.canvas.create_oval(
                        x1 + size / 2 - radius,
                        y1 + size / 2 - radius,
                        x1 + size / 2 + radius,
                        y1 + size / 2 + radius,
                        fill=LEGAL_MOVE,
                        outline="",
                    )

    def resign(self) -> None:
        if self.board.move_stack and not self.board.is_game_over():
            self.status_var.set("あなたが投了しました。AIの勝ちです。")
            self.show_result_overlay(None, resigned=True)

    def show_result_overlay(
        self, outcome: chess.Outcome | None, *, resigned: bool = False
    ) -> None:
        """対局結果をウィンドウ全体に一度だけ大きく表示する。"""
        if self.result_shown:
            return
        self.result_shown = True

        if resigned:
            title = "あなたの負け"
            detail = "投了しました"
            result_text = "AIの勝ち"
        elif outcome is None:
            title = "対局終了"
            detail = "結果を取得できませんでした"
            result_text = self.board.result()
        else:
            if outcome.winner is None:
                title = "引き分け"
            elif outcome.winner == self.human_color:
                title = "あなたの勝ち！"
            else:
                title = "あなたの負け"
            reasons = {
                chess.Termination.CHECKMATE: "チェックメイト",
                chess.Termination.STALEMATE: "ステイルメイト",
                chess.Termination.INSUFFICIENT_MATERIAL: "戦力不足",
                chess.Termination.SEVENTYFIVE_MOVES: "75手ルール",
                chess.Termination.FIVEFOLD_REPETITION: "同一局面5回",
                chess.Termination.FIFTY_MOVES: "50手ルール",
                chess.Termination.THREEFOLD_REPETITION: "同一局面3回",
                chess.Termination.VARIANT_WIN: "勝利条件達成",
                chess.Termination.VARIANT_LOSS: "敗北条件成立",
                chess.Termination.VARIANT_DRAW: "引き分け条件成立",
            }
            detail = reasons.get(outcome.termination, outcome.termination.name)
            result_text = self.board.result()

        overlay = tk.Frame(self.root, bg="#17212b")
        overlay.place(x=0, y=0, relwidth=1, relheight=1)
        overlay.lift()
        self.result_overlay = overlay

        content = tk.Frame(overlay, bg="#17212b")
        content.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(
            content,
            text="対局終了",
            font=("Yu Gothic UI", 20, "bold"),
            fg="#b8c4ce",
            bg="#17212b",
        ).pack(pady=(0, 18))
        tk.Label(
            content,
            text=title,
            font=("Yu Gothic UI", 42, "bold"),
            fg="#ffffff",
            bg="#17212b",
        ).pack()
        tk.Label(
            content,
            text=f"{detail}\n{result_text}",
            font=("Yu Gothic UI", 17),
            justify="center",
            fg="#d7e0e7",
            bg="#17212b",
        ).pack(pady=(18, 32))
        tk.Button(
            content,
            text="盤面に戻る",
            command=self.close_result_overlay,
            font=("Yu Gothic UI", 13, "bold"),
            bg="#f0d9b5",
            activebackground="#ffffff",
            relief="flat",
            padx=30,
            pady=10,
            cursor="hand2",
        ).pack()

    def close_result_overlay(self) -> None:
        if self.result_overlay is not None:
            self.result_overlay.destroy()
            self.result_overlay = None

    def on_close(self) -> None:
        self.save_settings()
        self.stop_engine()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    ChessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
