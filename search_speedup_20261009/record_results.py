from pathlib import Path
import json,statistics
work=Path(__file__).resolve().parent
report=json.loads((work/'search_validation.json').read_text())
rows=report['results']
before=statistics.median(row['baseline_seconds'] for row in rows)
after=statistics.median(row['candidate_seconds'] for row in rows)
table='\n## 最終測定と反映結果\n\n| 測定 | 変更前 | 変更後 |\n|---|---:|---:|\n'
for row in rows:table+=f"| {row['repeat']+1}回目 | {row['baseline_seconds']:.3f}秒 | {row['candidate_seconds']:.3f}秒 |\n"
table+=f'| 中央値 | {before:.3f}秒 | {after:.3f}秒 |\n\n中央値比で{before/after:.3f}倍、所要時間は{(1-after/before)*100:.1f}%短縮。1回目は変更後が遅く、短いGPU測定の変動を含む。実サイクルの短縮率は未測定。すべての測定で探索結果は完全一致した。\n\n正本4ファイルへ反映済み。反映前ソース一致、バックアップ作成、Python構文確認、ハッシュ記録が完了。\n'
path=work/'README.md';s=path.read_text();s=s.replace(' に反映する。',' に反映済み。');path.write_text(s+table)
project=Path('/home/user/chess/alphazero_chess_v2')
readme=project/'README.md'
addition='\n\n## 2026-10-09 探索処理の追加高速化\n\n盤面配列再利用、合法手生成共有、終局ノードの結果保存、候補比較簡略化、手番号表引き、推論入力バッファ再利用を反映。GUIからの実行方法は同じ。新しく起動したプロセスに適用される。8局面×800探索の3回測定は中央値5.879秒→5.169秒（約12%短縮）、全探索結果一致。実際の1サイクル全体の速度は未測定。詳細・再現条件は `C:/Users/User/Documents/ChatGPT/研究室/search_speedup_20261009/README.md`。\n'
with readme.open('a') as f:f.write(addition)
print(json.dumps(dict(median_before=before,median_after=after,speedup=before/after,reduction_percent=(1-after/before)*100)))
