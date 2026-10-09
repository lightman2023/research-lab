from pathlib import Path

source=Path(__file__).with_name('analyze_evaluation.py').read_text()
source=source.replace('match-20261005-090556-8608c126.pgn','match-20261005-131609-7c4782ef.pgn')
start=source.index("for side,branch in")
end=source.index('\ngames=[]',start)
source=source[:start]+'''assert metadata['device']=='cpu'
for side,expected in [('a',project/'experiments/live-lr0001-temp1-20261002/checkpoints/saved/cycle-000010.msgpack'),('b',project/'experiments/search-budget-20261004/B200/checkpoints/saved/cycle-000005.msgpack')]:
    assert Path(metadata[f'model_{side}']).resolve()==expected.resolve()
    assert metadata[f'model_{side}_sha256']==hashlib.sha256(expected.read_bytes()).hexdigest()
'''+source[end:]
source=source.replace('evaluation_summary.json','baseline_cpu_evaluation_summary.json')
source=source.replace('Inconclusive strength difference; repetition remains; improvement over starting model untested.','B won 6, baseline won 2, draws 4. Favorable evidence under this small paired test; repetition remains.')
exec(compile(source,__file__,'exec'))
