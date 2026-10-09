# Chess self-play training

WSL Ubuntuで動くチェスAIの自己対局・学習コードです。`alphazero_chess` が従来版v1、`alphazero_chess_v2` が縮小AlphaZero型v2です。v2のWindows GUIは別リポジトリ `chess-v2-windows` にあります。

## 構成

- `alphazero_chess/`：v1のPolicy・Valueネットワーク、MCTS、自己対局、学習、対戦、モデル比較
- `alphazero_chess_v2/`：v2の119面入力、Residual Policy・Valueネットワーク、MCTS、自己対局、学習、モデル比較
- ルートのスクリプト：Dala/Lc0を試した初期の補助コード。Dala本体や学習済みモデルは含みません。

## v2を新規に試す

UbuntuのPython 3.11以降とCUDA対応環境を用意し、`alphazero_chess_v2` に移動して依存関係を `pyproject.toml` からインストールしてください。例：

```bash
cd alphazero_chess_v2
uv sync
uv run python init_model.py
uv run python selfplay.py --games 1 --simulations 50
uv run python train.py --steps 64
```

v1の使い方は [`alphazero_chess/README.md`](alphazero_chess/README.md) を参照してください。両者は別のチェックポイントと自己対局データを使います。

## 公開範囲

ソースコードと依存関係の定義だけを含めています。学習済みチェックポイント、自己対局データ、棋譜、ログ、仮想環境、外部から取得したDalaソースやモデルは含めません。チェックポイントがない状態では `init_model.py` から初期化できます。実運用に使ったWindows GUIは固定パスを前提としているため、別の場所に配置する場合はパス設定の変更が必要です。
