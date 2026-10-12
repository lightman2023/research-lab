import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json,importlib.util,importlib,time
from pathlib import Path
import chess,chess.pgn,numpy as np
work=Path(__file__).resolve().parent
for label in ['baseline','candidate']:
    spec=importlib.util.spec_from_file_location(label,work/label/'azchess/__init__.py',submodule_search_locations=[str(work/label/'azchess')])
    module=importlib.util.module_from_spec(spec);sys.modules[label]=module;spec.loader.exec_module(module)
old=importlib.import_module('baseline.game');new=importlib.import_module('candidate.history')
project=Path('/home/user/chess/alphazero_chess_v2');cases=0
start=time.perf_counter()
for filename in ['match-20261007-085457-07445777.pgn','match-20261007-132552-48097e54.pgn']:
    with (project/'arena_results'/filename).open() as f:
        while (game:=chess.pgn.read_game(f)) is not None:
            b=game.board();encoder=new.HistoryEncoder(b)
            for i,move in enumerate(game.mainline_moves()):
                if i%5==0:
                    np.testing.assert_array_equal(encoder.encode(),old.encode_board(b));assert encoder.outcome()==old.game_outcome(b);cases+=1
                b.push(move);encoder.push()
            np.testing.assert_array_equal(encoder.encode(),old.encode_board(b));assert encoder.outcome()==old.game_outcome(b);cases+=1
            # Restore the full tree path and validate prefixes while unwinding.
            while b.move_stack:
                b.pop();encoder.pop()
                if len(b.move_stack)%17==0:
                    np.testing.assert_array_equal(encoder.encode(),old.encode_board(b));assert encoder.outcome()==old.game_outcome(b);cases+=1
for fen in ['r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1','4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1','4k3/P7/8/8/8/8/7p/4K3 w - - 0 1','7k/5K2/6R1/8/8/8/8/8 w - - 99 1','8/8/8/8/8/8/4k3/7K w - - 150 1']:
    b=chess.Board(fen);encoder=new.HistoryEncoder(b)
    np.testing.assert_array_equal(encoder.encode(),old.encode_board(b));assert encoder.outcome()==old.game_outcome(b);cases+=1
    for move in list(b.legal_moves):
        before=b.fen();b.push(move);encoder.push()
        np.testing.assert_array_equal(encoder.encode(),old.encode_board(b));assert encoder.outcome()==old.game_outcome(b);cases+=1
        b.pop();encoder.pop();assert b.fen()==before
result=dict(cases=cases,input_exact=True,outcomes_exact=True,seconds=time.perf_counter()-start,chess_version=chess.__version__)
(work/'history_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
