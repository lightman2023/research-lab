import sys,importlib,importlib.util,json
from pathlib import Path
import chess
work=Path(__file__).resolve().parent
for label in ('baseline','candidate'):
    spec=importlib.util.spec_from_file_location(label,work/label/'azchess/__init__.py',submodule_search_locations=[str(work/label/'azchess')])
    module=importlib.util.module_from_spec(spec);sys.modules[label]=module;spec.loader.exec_module(module)
old=importlib.import_module('baseline.game');new=importlib.import_module('candidate.game')
cases=0
for turn in (chess.WHITE,chess.BLACK):
    board=chess.Board();board.turn=turn
    for source in range(64):
        for target in range(64):
            for promotion in (None,chess.QUEEN,chess.KNIGHT,chess.BISHOP,chess.ROOK):
                move=chess.Move(source,target,promotion=promotion)
                try:reference=old.move_to_action(board,move)
                except ValueError:
                    try:new.move_to_action(board,move)
                    except ValueError:pass
                    else:raise AssertionError(move)
                else:assert new.move_to_action(board,move)==reference
                cases+=1
report=dict(cases=cases,action_indices_exact=True,invalid_moves_rejected=True)
(work/'action_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
