"""Python環境とDalaモデルの準備状態を確認する。"""

from pathlib import Path

import chess

project_dir = Path(__file__).resolve().parent
models = list((project_dir / "models").glob("*900*.pb.gz"))

print(f"python-chess: OK ({chess.__version__})")
print(f"Dala-900: {'OK - ' + str(models[0]) if models else '未取得'}")

