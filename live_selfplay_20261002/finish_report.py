from pathlib import Path
import json
import shutil
import hashlib
import numpy as np
ROOT=Path('/home/user/chess/alphazero_chess_v2/experiments/live-lr0001-20261002-run2')
WORK=Path('/mnt/c/Users/User/Documents/ChatGPT/研究室/live_selfplay_20261002')
data=json.loads((ROOT/'summary.json').read_text())
stats=json.loads((ROOT/'analysis.json').read_text())
prefixes=[]
for i in range(1,9):
    with np.load(ROOT/f'temperature-0.25/game-{i:04d}.npz') as a, np.load(ROOT/f'temperature-1.0/game-{i:04d}.npz') as b:
        n=min(30,len(a['moves']),len(b['moves']))
        assert np.array_equal(a['moves'][:n],b['moves'][:n])
        prefixes.append(n)
stats['paired_prefixes_equal_until_ply30']=prefixes
(ROOT/'analysis.json').write_text(json.dumps(stats,indent=2))
for name in ['run_live.py','test_core.py','core_test_results.json','inspect_collapse.py','analyze_results.py','check_gui_listing.py','gui_listing_results.json','prepare.py','finish_configuration.py','update_gui.py','finish_report.py']:
    shutil.copy2(WORK/name,ROOT/name)
inspection=json.loads((ROOT/'collapse_inspection.json').read_text())
inspection['provenance']=dict(checkpoint=str(ROOT/'checkpoints/saved/cycle-000002.msgpack'),data='first 16 game files',sample_positions=128,seed=20261002,method='same inputs compared with inference BN and batch BN; value-loss-only gradients in train mode')
(ROOT/'collapse_inspection.json').write_text(json.dumps(inspection,indent=2))
latest=ROOT/'checkpoints/latest.msgpack'
final=ROOT/'checkpoints/saved/cycle-000010.msgpack'
assert hashlib.sha256(latest.read_bytes()).digest()==hashlib.sha256(final.read_bytes()).digest()
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ['.msgpack','.npz','.pgn','.py']}
(ROOT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2))
cycle_rows='\n'.join(f"| {r['cycle']} | {r['terminations'].get('THREEFOLD_REPETITION',0)} | {r['terminations'].get('CHECKMATE',0)} | {r['evaluation']['dead']}/129 | {r['live_training_sample']['dead']}/128 | {r['evaluation']['value_mse']:.6f} |" for r in data['history'])
report=f'''# 自己対局学習と反復の検証（2026-10-02）

## 結論

1〜3の作業は完了しました。10サイクルの学習で、検査した固定129局面では広範な価値予測の固定化は再発しませんでした。しかし、従来の手選択（後半温度0.25）では反復が多く残っています。
同じ最終モデルで温度だけを変えた比較では、温度1は8局中5局がチェックメイト、反復引き分け0局でした。往復の割合は下がりましたが、対局は長くなり、往復自体も残っています。強さの改善は未測定です。

## 1. 基本動作と終局ルール

- 対象の正本：`/home/user/chess/alphazero_chess_v2`。Windows側はモデル比較GUIを更新。
- `game_outcome()` を探索・自己対局・arenaで共用。実際の3回反復、進展なし100半手で自動引き分け。次の手で請求可能になるだけの局面は終局扱いにしない。
- 引き分け請求の追加アクションは除去。詰みは50手ルールより先に判定。
- 詰みの勝敗・符号、MCTSのバックアップと詰み手の選択、反復履歴の入力・探索への継承、手の対応、白黒の昇格・成り分け、キャスリング、アンパッサン、加工前の訪問割合の学習対象を確認。22項目成功。無作為局面を含む25,336合法手の対応も検査。
- 探索の論理テストではルールから作った局面と一様方策のテスト用探索器を使用。実モデルの全局面での詰ませる能力を保証する検査ではない。テスト局面は学習データに入れていない。
- 変更前の5ファイルを `.pre-azrules-20261002` で保存。人間対戦サーバーの請求UIは今回の対象外。

## 2. 低学習率で新しい自己対局データを作る学習

開始モデルは前回の学習率比較B（0.0001）の1600更新版。元の50サイクルモデルから派生したモデルです。AdamWの状態と更新回数を引き継ぎました。一定学習率の保存状態を減衰スケジュールの形式で読もうとして初回の準備が失敗したため、対応する一定学習率の形式に修正しています。失敗した準備フォルダーも保持し、そこでは対局・学習は行っていません。

設定：10サイクル、1サイクル8局・64更新、計80局・640追加更新。学習率0.0001、AdamW、重み減衰0.0001、バッチ64、探索50回、最大512半手。0〜29半手は温度1、30半手以降は0.25。根のDirichletノイズはα=0.3、混合率0.25。古いルールのデータは混ぜず、この実験で作ったデータだけを使用。最新モデルと状態を各サイクルで更新、世代モデルも各サイクル保存しました。保存間隔はランナーの `--save-interval` で指定可能です。

最初の2局で保存・手の分布を確認し、最初の64更新で有限な損失・価値予測を確認してから継続しました。seed基準は20261002、各対局は基準+対局番号、各サイクルの抽出は基準+1000+サイクル番号です。最終更新回数は2240です。

| サイクル | 反復引き分け | 詰み | 固定集合の不活性数 | 自己対局抽出の不活性数 | 固定集合の価値MSE |
|---|---:|---:|---:|---:|---:|
{cycle_rows}

不活性は価値ヘッドのDense_0の256成分が全て非正で、直後のReLU出力が全て0になる状態を指します。固定集合は初期局面3、引き分け対局由来64、勝ち側31、負け側31。MSEには初期局面3を含めません。ラベルは実際の自己対局結果であり、局面の客観的な必勝・必敗を示すラベルではありません。固定集合は古いルールの対局から作った診断集合なので、新ルールでの一般化性能を直接測るものでもありません。

80局の結果：3回反復67局、チェックメイト13局、手数上限の打ち切り0局。平均71.61半手。学習用5729局面の結果ラベルは0が4937、+1が398、−1が394。価値MSEは開始0.481293から最終0.474317へ小幅に減少しましたが、勝ち側MSEはなお0.9356、負け側は0.9779です。勝敗を十分に予測できると断定しません。

### 途中停止の追加診断

2サイクル目で自己対局の抽出128局面中1局面に不活性を検出し、当初の「1件で停止」の条件で止まりました。別の128局面では推論時2件、学習時0件で、Dense_0・Dense_1の価値損失の勾配も非ゼロでした。固定129局面には0件で、広範な固定化と判断する根拠はありませんでした。
停止条件を検査集合の10%以上への拡大に変更し、同じモデル・学習状態から再開しました。これは観察後に変更した診断用の監視条件で、事前登録した比較実験ではありません。2サイクル時点の結果・ソースを別名で保持し、`protocol_amendment.json` に変更を記録。後のサイクルで不活性は固定集合・抽出集合とも0件でした。この短期結果だけで将来の再発が防げるとは言えません。

## 3. 結果に応じた手選択の比較

広範な固定化がなく、反復が残ったため、最終モデルを固定し、30半手以降の温度だけを0.25と1に変えて各8局を自己対局させました。両側が同じモデルで指す比較であり、モデル同士の強さ比較ではありません。探索50回・ノイズ・上限512半手は同じ。8組で乱数seedが一致し、30半手までの手順も全て一致しました。seedは20271002〜20271009。比較用16局は学習に使用していません。

| 条件 | 反復引き分け | チェックメイト | 50手ルール | 駒不足 | 上限打ち切り | 平均半手 | 往復割合 | 往復総数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 温度0.25 | 8 | 0 | 0 | 0 | 0 | 77.875 | 18.12% | 110 |
| 温度1 | 0 | 5 | 2 | 1 | 0 | 240.875 | 8.69% | 166 |

往復は、自分の直前の手の出発点・到達点を逆にする手を数えています。他の形の循環を全て検出する指標ではありません。温度1では対局が長いため、割合は減っても往復の総数は増えました。完全に往復をなくせた、あるいは勝ち切る能力が上がった、とは言えません。

訪問回数をN、温度をTとすると、実際に選ぶ手の確率は N^(1/T) に比例します。20訪問と10訪問の2手だけなら、T=1では2:1（約67%:33%）、T=0.25では16:1（約94%:6%）。後半の温度0.25は多く訪問した手へ強く集中させます。今回の8組では、その集中を緩めると反復終局を避けて通常の詰み・引き分けに進むことを確認しました。より広い条件での再現性は未確認です。学習対象はどちらもN/総訪問数であり、反復禁止・引き分け罰・駒得による仮の勝敗は入れていません。

## 検証・成果物・今後

96局全てで棋譜の局数・手数、保存状態と履歴、最初の終局タイミング、終局理由、各局面の勝敗ラベルを照合し成功。元の開始モデル・学習状態のハッシュは不変。最新モデルと10サイクル保存モデルのハッシュも一致。終了時に自己対局・学習・arenaの実験プロセスは残っていません。

Windowsの `C:/Users/User/Desktop/chess_v2/v2モデル比較を起動.bat` の「低学習率・自己対局」で保存モデルを選べます。cycle-000000が開始モデル、cycle-000010が今回の最終モデル。モデル一覧取得と実験中のGPU競合検出を検査しました。今回、対局GUIを実際に操作して長い強さ比較を実行したわけではありません。

次の候補は、学習率0.0001・温度1で新しい自己対局学習を短期間続け、終局理由・価値予測・強さを確認することです。今回の追加学習は温度0.25で行ったので、温度1による学習後の改善はまだ検証していません。原論文に寄せるなら温度1が候補ですが、今回の8組だけで本番設定の最適性は断定しません。各試験の局数は小さく、学習率についても新ルールの生きた自己対局で高学習率との並行比較はしていません。

モデル・状態・データ・棋譜・ソース・ログ・設定・ハッシュはこのフォルダーに保存。`summary.json`、`analysis.json`、`core_test_results.json`、`collapse_inspection.json`、`artifact_hashes.json` を参照。再現用の初回コマンドは、プロジェクト内のPythonで `run_live.py --output-dir NEW_DIRECTORY --cycles 10`。同じ診断手順の途中再開は `--resume` を使います。一定学習率の状態を通常のtrain.pyで継続するときは `--learning-rate 0.0001 --constant-learning-rate` が必要です。通常の古い学習GUIを新実験の継続として使わないでください。

## 論文との関係

David Silver, Thomas Hubert, Julian Schrittwieser ほか（2017）, *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*, https://arxiv.org/html/1712.01815v1 。本文の式(1)、Methods「Representation」「Configuration」を確認しました。実際の3回反復・50手による自動引き分け、訪問回数に比例する自己対局の手選択、実際の終局結果と探索分布を学習対象にする説明が根拠です。
今回の学習率0.0001、AdamW、探索50回、ネットワーク規模、バッチ64、監視の10%という閾値はこの実装・独自実験の設定です。原論文は学習時800探索であり、今回の縮小版の性能を保証するものではありません。
'''
(ROOT/'README.md').write_text(report,encoding='utf-8')
(WORK/'report.md').write_text(report,encoding='utf-8')
for name in ['summary.json','analysis.json','configuration.json','protocol_amendment.json','collapse_inspection.json','artifact_hashes.json','run.log']:
    shutil.copy2(ROOT/name,WORK/name)
gui_readme=Path('/mnt/c/Users/User/Desktop/chess_v2/README.md')
text=gui_readme.read_text(encoding='utf-8')
if '2026-10-02 低学習率・自己対局モデル' not in text:
    text+='\n\n## 2026-10-02 低学習率・自己対局モデル\n\n`v2モデル比較を起動.bat` の「低学習率・自己対局」で新実験の保存モデルを選べます。cycle-000000は開始モデル、cycle-000010は10サイクル後です。探索・自己対局・arenaは実際の3回反復と50手ルールで自動引き分けに統一しています。自己対局・学習とモデル比較は同時に実行できません。新実験の継続学習は学習率0.0001の一定設定が必要です。従来の学習GUIはその設定に対応していないため、この実験の継続には使わないでください。\n'
    gui_readme.write_text(text,encoding='utf-8')
print(json.dumps(dict(report=str(WORK/'report.md'),validated=96,paired_prefixes=prefixes,checkpoints=len(list((ROOT/'checkpoints/saved').glob('*.msgpack'))),hash_files=len(hashes))))
