"""ターミナル上でDala-900と対局する。指し手は e2e4 形式で入力する。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import chess.engine


def find_model(project_dir: Path) -> Path:
    models = sorted((project_dir / "models").glob("*900*.pb.gz"))
    if not models:
        raise FileNotFoundError("Dala-900がありません。先に python download_dala.py を実行してください。")
    return models[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dala-900とチェスをします")
    parser.add_argument("--lc0", required=True, type=Path, help="lc0.exeの場所")
    parser.add_argument("--color", choices=("white", "black", "random"), default="white")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    lc0 = args.lc0.expanduser().resolve()
    model = find_model(project_dir)
    if not lc0.is_file():
        raise FileNotFoundError(f"lc0.exeが見つかりません: {lc0}")

    color = random.choice(("white", "black")) if args.color == "random" else args.color
    human_color = chess.WHITE if color == "white" else chess.BLACK
    board = chess.Board()

    print(f"あなたは{'白' if human_color else '黒'}です。終了するには quit と入力します。")
    engine = chess.engine.SimpleEngine.popen_uci([str(lc0), f"--weights={model}"])
    try:
        # Dalaは浅い探索で人間らしい棋力を出すモデル。利用可能なら温度も設定する。
        if "Temperature" in engine.options:
            engine.configure({"Temperature": 1.0})

        while not board.is_game_over():
            print("\n" + str(board) + "\n")
            if board.turn == human_color:
                move_text = input("あなたの指し手 (例 e2e4): ").strip().lower()
                if move_text in {"quit", "exit"}:
                    print("対局を終了しました。")
                    return
                try:
                    move = chess.Move.from_uci(move_text)
                except ValueError:
                    print("形式が違います。e2e4のように入力してください。")
                    continue
                if move not in board.legal_moves:
                    print("その手は指せません。")
                    continue
            else:
                print("Dala-900が考えています...")
                move = engine.play(board, chess.engine.Limit(nodes=1)).move
                print(f"Dala-900: {move.uci()}")
            board.push(move)

        print("\n" + str(board))
        print(f"\n対局結果: {board.result()} ({board.outcome().termination.name})")
    finally:
        engine.quit()


if __name__ == "__main__":
    main()

