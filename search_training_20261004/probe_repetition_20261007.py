import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json,hashlib,argparse
from pathlib import Path
from collections import Counter
PROJECT=Path('/home/user/chess/alphazero_chess_v2');sys.path.insert(0,str(PROJECT))
import chess,chess.pgn
from azchess.mcts import MCTS
from azchess.checkpoint import load_variables
from azchess.game import game_outcome
class Probe(MCTS):
    def search(self,board):
        self.root_ply=len(board.move_stack);self.draws=Counter()
        return super().search(board)
    def expand(self,node,board):
        if len(board.move_stack)>self.root_ply:
            outcome=game_outcome(board)
            if outcome and outcome.winner is None:self.draws[board.move_stack[self.root_ply].uci()]+=1
        return super().expand(node,board)
pgn=PROJECT/'arena_results/match-20261007-085457-07445777.pgn'
parser=argparse.ArgumentParser();parser.add_argument('--deep',action='store_true');args=parser.parse_args()
sims=800 if args.deep else 200
meta=json.loads(pgn.with_suffix('.json').read_text());agents={}
for side in ['A','B']:
    path=Path(meta['model_'+side.lower()]);assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['model_'+side.lower()+'_sha256']
    model,variables=load_variables(path);agents[side]=Probe(model,variables,simulations=sims)
games=[]
with pgn.open() as f:
    while (g:=chess.pgn.read_game(f)) is not None:games.append(g)
out=Path(__file__).with_name('repetition_probe_20261007_deep.json' if args.deep else 'repetition_probe_20261007_results.json')
result=dict(pgn=str(pgn),models={s:meta['model_'+s.lower()] for s in ['A','B']},simulations=sims,root_noise=False,device='cpu',selection='B at games 16/22/24, offsets 1/2/1' if args.deep else 'last two plies before each of nine repetition draws',records=[])
for number,g in enumerate(games,1):
    if g.headers['Termination']!='THREEFOLD_REPETITION':continue
    moves=list(g.mainline_moves())
    for offset in [2,1]:
        if args.deep and (number,offset) not in [(16,1),(22,2),(24,1)]:continue
        ply=len(moves)-offset;board=g.board()
        for move in moves[:ply]:board.push(move)
        side=g.headers['White' if board.turn else 'Black'].split(':')[0];agent=agents[side]
        assert game_outcome(board) is None
        _,value=agent.evaluate(board);root=agent.search(board);rows=[]
        for move,child in root.children.items():
            b=board.copy(stack=True);b.push(move);outcome=game_outcome(b)
            rows.append(dict(uci=move.uci(),san=board.san(move),visits=child.visit_count,prior=child.prior,
                q=-child.value if child.visit_count else None,termination=outcome.termination.name if outcome else None,
                draw_leaves=agent.draws[move.uci()],reverse=ply>=2 and move.from_square==moves[ply-2].to_square and move.to_square==moves[ply-2].from_square))
        rows.sort(key=lambda r:-r['visits'])
        record=dict(game=number,offset=offset,ply=ply,side=side,fen=board.fen(),history=[m.uci() for m in moves[:ply]],in_check=board.is_check(),legal_moves=len(rows),value=value,actual=moves[ply].uci(),selected=rows[0]['uci'],moves=rows)
        result['records'].append(record);out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        alternatives=[r for r in rows if not r['reverse'] and not r['termination'] and r['visits']]
        best=max(alternatives,key=lambda r:r['q']) if alternatives else None
        print(json.dumps(dict(game=number,side=side,offset=offset,actual=record['actual'],top=rows[0],best_nonreverse=best),ensure_ascii=False),flush=True)
print('DONE',len(result['records']),flush=True)
