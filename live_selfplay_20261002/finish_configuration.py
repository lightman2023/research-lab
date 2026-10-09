from pathlib import Path
root=Path('/home/user/chess/alphazero_chess_v2')
path=root/'train.py'
text=path.read_text()
text=text.replace('def create_optimizer(learning_rate: float = 1e-3):', 'def create_optimizer(learning_rate: float = 1e-3, constant: bool = False):\n    if constant:\n        return optax.adamw(learning_rate, weight_decay=1e-4)')
text=text.replace('    parser.add_argument("--seed", type=int)', '    parser.add_argument("--seed", type=int)\n    parser.add_argument("--constant-learning-rate", action="store_true", help="Use for optimizer states from constant-rate experiments")')
text=text.replace('optimizer = create_optimizer(args.learning_rate)', 'optimizer = create_optimizer(args.learning_rate, constant=args.constant_learning_rate)')
path.write_text(text)
readme=root/'README.md'
text=readme.read_text()
text+='''

## 2026-10-02 終局ルールと低学習率の自己対局実験

`azchess.game.game_outcome()` を探索・自己対局・モデル比較で共用します。
実際の3回反復、100半手（双方50手）の進展なしで自動引き分けとし、
次の手を指すと請求できるだけの局面では終了しません。詰み等を先に判定します。
ルール識別子は `az_actual_threefold_50move_v1`。旧棋譜・データは旧ルールの記録です。
GUIの人間対戦サーバーは今回の対象外で、対人戦の請求UIは追加していません。

新実験は `experiments/live-lr0001-20261002-run2` に分離しています。
学習率比較Bの1600更新モデルとAdamW状態から開始し、新ルールで生成したデータだけを使用します。
通常の学習GUIは従来の学習率を使用します。新実験のモデル・状態をそこへ上書きしないでください。

`train.py --learning-rate 0.0001 --constant-learning-rate --seed SEED` で
一定学習率の状態を継続できます。`--constant-learning-rate` を省略すると既存の減衰スケジュールです。
一定学習率と減衰スケジュールでは保存状態の構造が異なるため、状態ファイルに合わせて指定してください。
`selfplay.py --late-temperature 1 --seed SEED` で30半手以降も訪問回数に比例して手を選びます。
学習対象の手の分布は温度にかかわらず加工前の訪問回数です。

Windowsの `v2モデル比較を起動.bat` の「低学習率・自己対局」で新実験の保存モデルを比較できます。
`cycle-000000` は開始モデルです。自己対局・学習中は比較GUIからの実行を止めます。
'''
readme.write_text(text)
print('Documented rules, experiment, constant optimizer resume and GUI model selection')
