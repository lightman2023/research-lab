# AlphaZero-style Chess Lab

これは学習教材用の小型AlphaZero型システムです。DeepMindの学習済みAlphaZeroではありません。
4 Residual Blocks・64 ChannelsのPolicy-Value Network、MCTS、自己対局、学習を含みます。

## 学習前のAIと対戦

```bash
cd ~/chess/alphazero_chess
uv run python init_model.py
uv run python play.py --color white --simulations 50
```

指し手は `e2e4` 形式です。初期モデルはランダム重みなので意図的に弱い状態です。

## 強化学習サイクル

```bash
uv run python selfplay.py --games 10 --simulations 50
uv run python train.py --epochs 3
```

自己対局と学習を繰り返します。最新モデルは `checkpoints/latest.msgpack` に保存されます。

