from pathlib import Path
import statistics,json
work=Path(__file__).resolve().parent
rows=json.loads((work/'search_validation.json').read_text())['results']
before=statistics.median(r['baseline_seconds'] for r in rows)
after=statistics.median(r['candidate_seconds'] for r in rows)
table='| 測定 | 変更前 | 変更後 |\n|---|---:|---:|\n'
for row in rows:table+=f"| {row['repeat']+1}回目 | {row['baseline_seconds']:.3f}秒 | {row['candidate_seconds']:.3f}秒 |\n"
table+=f'| 中央値 | {before:.3f}秒 | {after:.3f}秒 |\n'
text=f'''# 推論呼び出しと探索経路の追加高速化（2026-10-09）

正本 `/home/user/chess/alphazero_chess_v2` の、前回の追加高速化を反映済みのコードを比較元にする。

## 実装・採用した変更

- バッチ推論ではfloat32のNumPy配列をJIT関数へ直接渡す。毎回のjnp.asarray呼び出しを省き、転送はJAXに任せる。実際の入力形式と同じNumPy配列で、1・2・4・8などのサイズを事前準備する。GPU計算を終えたdevice_getの後でバッファを再利用する。
- 合法手の生成と終局判定を行った後、出力番号への対応表作成を遅延する。GPUへ推論を依頼してから、対応表と番号配列をCPUで作成する。対局ごとのコルーチンは結果待ちで停止しているため、この間の盤面は変化しない。既存の辞書形式の推論要求も受け付ける。
- 終局が確認済みの探索ノードを再訪した場合、その枝の最後の盤面push/popと入力履歴更新を省き、保存した価値をそのまま反映する。探索の経路と価値の符号反転、訪問数更新は従来と同じ。単独MCTSとバッチMCTSに適用する。

## 試して採用を見送った案

GPU側で合法手の出力だけを取り出し、全4,672候補から218枠へ戻すデータを減らす版を試した。固定入力8局面、200呼び出し、4反復で、通常の直接呼び出しより有利にならなかった。追加処理と番号配列転送が発生する。出力値は一致したが、正本には反映していない。結果はdispatch_validation.json/log、再現はtry_compact.py。

## 測定と確認結果

GPUバッチ比較はB50保存モデル、RTX 3060、8局面×800探索、8同時、ルートノイズあり、seed90100+番号。初期局面2つと実棋譜終局5半手前の6局面。各版3回、順番を交互にする。ウォームアップ後の探索を測定し、結果取得までを時間に含める。

{table}

中央値で約{(1-after/before)*100:.1f}%の時間短縮（速度比{before/after:.3f}倍）。短いGPU測定には変動があり、この結果から学習サイクル全体の速度や強さの改善は断定しない。

3反復すべて、探索木93,847ノードの訪問回数・価値合計・事前確率、子ノードの順番が完全一致。推論呼び出し数も一致。CPU単独200探索の初期局面と50手ルール直前のルーク終盤、計4,950ノードも一致。CPU自己対局2局、2同時、64探索、12半手、seed90110/90111を新規生成し、保存NPZの全項目が一致。短い打ち切りは検証用ディレクトリだけで使用する。

search_validation.json/logに最終測定、single_validation.jsonとselfplay_validation.jsonに確認結果、baseline_profile.txt/candidate_profile.txtに200探索の処理内訳を保存する。プロファイルには測定器の負荷を含むため、速度比較には通常測定を使う。

## 反映と操作

正本のazchess/batched.py、mcts.pyに反映。GUIは従来の実行ボタンを使う。新しく起動する自己対局・評価プロセスから適用される。読み込み済みのプロセスは次回起動から適用される。

反映前に正本とbaselineのソース一致を確認し、.pre-inference-speedup-20261009のバックアップを作成する。Python構文と反映ハッシュはdeployment.jsonに記録。モデルと既存データは操作しない。

再現はWSL正本の.venv/bin/pythonでverify_search.py、verify_single.py、verify_selfplay.pyを実行する。自己対局試験は保存済みシードを再利用するので、新しく生成し直す際は検証用保存先を変更する。

## 根拠と残る課題

JAX公式 [Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html) の本文は前の調査で確認済み。ホストへ値を取り出す前にはPythonを進められる性質を利用する。今回の個別の処理変更と効果は実コードの検証で確認した。

学習方法の基準はDavid Silver、Thomas Hubert、Julian Schrittwieserほか（2017）、[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/html/1712.01815v1)、本文の探索説明と式(1)。本文は前の調査で確認済み。今回の高速化は独自の実装上の変更であり、論文が個別に指定した最適化と扱わない。

残る大きな候補はCPU探索グループとGPU推論をさらに重ねる構造変更、完全な入力一致に基づく推論キャッシュ、探索の低水準言語化。実対局全体の時間を確認してから優先順位を決める。
'''
(work/'README.md').write_text(text)
readme=Path('/home/user/chess/alphazero_chess_v2/README.md')
with readme.open('a') as f:
    f.write(f'\n\n## 2026-10-09 推論呼び出しと探索経路の追加高速化\n\nバッチ推論の直接入力、GPU計算中の合法手対応表作成、終局済みノードの盤面更新省略を反映。8局面×800探索の中央値{before:.3f}秒→{after:.3f}秒。探索木と保存データの一致を確認。GUI操作は同じ。新規起動プロセスから適用。詳細は `C:/Users/User/Documents/ChatGPT/研究室/inference_speedup_20261009/README.md`。\n')
print(json.dumps(dict(median_before=before,median_after=after,reduction_percent=(1-after/before)*100)))
