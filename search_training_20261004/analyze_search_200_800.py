from pathlib import Path
source=Path(__file__).with_name('analyze_comparison_20261007.py').read_text()
source=source.replace('match-20261007-085457-07445777.pgn','match-20261007-132552-48097e54.pgn')
source=source.replace("meta['simulations_a']==meta['simulations_b']==200", "meta['simulations_a']==200 and meta['simulations_b']==800")
source=source.replace("for side in ['a','b']:", "assert meta['model_a']==meta['model_b'] and meta['model_a_sha256']==meta['model_b_sha256']\nfor side in ['a','b']:")
source=source.replace('comparison_20261007_summary.json','search_200_800_20261007_summary.json')
exec(compile(source,__file__,'exec'))
