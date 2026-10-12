from pathlib import Path
work=Path(__file__).resolve().parent
path=work/'candidate/azchess/game.py'
s=path.read_text()
original=(work/'baseline/azchess/game.py').read_text()
start=original.index('def move_to_action(');end=original.index('\n\ndef legal_action_map',start)
fallback=original[start:end].replace('def move_to_action(', 'def _uncached_move_to_action(')
s=s.replace('    if action < 0:\n        raise ValueError(f"Unencodable move: {move}")', '    if action < 0:\n        return _uncached_move_to_action(board, move)')
s=s.replace('\n\ndef legal_action_map', '\n\n'+fallback+'\n\ndef legal_action_map')
path.write_text(s)
print('Preserved legacy handling of non-chess geometries')
