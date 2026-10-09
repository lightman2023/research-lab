import json
from pathlib import Path
import chess
path=Path(__file__).with_name('repetition_probe_20261007_results.json');data=json.loads(path.read_text())
assert len(data['records'])==18
summary=[]
for r in data['records']:
    board=chess.Board(r['fen']);own=board.turn
    pieces=lambda color:''.join(chess.piece_symbol(p).upper()*len(board.pieces(p,color)) for p in range(1,7))
    actual=next(m for m in r['moves'] if m['uci']==r['actual']);visited=[m for m in r['moves'] if m['visits']]
    alternatives=[m for m in visited if not m['reverse'] and m['termination'] is None]
    best=max(alternatives,key=lambda m:m['q']) if alternatives else None
    summary.append(dict(game=r['game'],side=r['side'],offset=r['offset'],in_check=r['in_check'],own_pieces=pieces(own),opponent_pieces=pieces(not own),actual=actual,best_nonreverse=best,reproduced=r['selected']==r['actual'],visited_moves=len(visited),legal_moves=r['legal_moves']))
output=dict(selection=data['selection'],records=summary,by_side={s:dict(positions=sum(r['side']==s for r in summary),reproduced=sum(r['side']==s and r['reproduced'] for r in summary),higher_q_alternative=sum(r['side']==s and r['best_nonreverse'] is not None and r['best_nonreverse']['q']>r['actual']['q'] for r in summary)) for s in ['A','B']})
path.with_name('repetition_probe_20261007_summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
print(json.dumps(output,ensure_ascii=False))
