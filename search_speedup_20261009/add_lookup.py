from pathlib import Path
work=Path(__file__).resolve().parent
path=work/'candidate/azchess/game.py'
s=path.read_text()
start=s.index('def move_to_action(')
end=s.index('\n\ndef legal_action_map',start)
s=s[:start]+'''def _build_normal_actions():
    table = [[-1]*64 for _ in range(64)]
    for source in range(64):
        for plane, (df, dr) in enumerate(QUEEN_DIRECTIONS):
            for distance in range(1, 8):
                file, rank = (source & 7)+df*distance, (source >> 3)+dr*distance
                if 0 <= file < 8 and 0 <= rank < 8:
                    table[source][rank*8+file] = source*ACTION_PLANES+plane*7+distance-1
        for plane, (df, dr) in enumerate(KNIGHT_DIRECTIONS):
            file, rank = (source & 7)+df, (source >> 3)+dr
            if 0 <= file < 8 and 0 <= rank < 8:
                table[source][rank*8+file] = source*ACTION_PLANES+56+plane
    return tuple(tuple(row) for row in table)

_NORMAL_ACTIONS = _build_normal_actions()
_UNDERPROMOTION_INDEX = {piece: i for i, piece in enumerate(UNDERPROMOTIONS)}

def move_to_action(board: chess.Board, move: chess.Move) -> int:
    flip = 0 if board.turn == chess.WHITE else 56
    source, target = move.from_square ^ flip, move.to_square ^ flip
    promotion = _UNDERPROMOTION_INDEX.get(move.promotion)
    if promotion is not None:
        df, dr = (target & 7)-(source & 7), (target >> 3)-(source >> 3)
        if dr != 1 or df not in (-1, 0, 1):
            raise ValueError(f"Invalid underpromotion: {move}")
        return source*ACTION_PLANES+64+(df+1)*3+promotion
    action = _NORMAL_ACTIONS[source][target]
    if action < 0:
        raise ValueError(f"Unencodable move: {move}")
    return action
'''+s[end:]
path.write_text(s)
path=work/'candidate/azchess/batched.py'
s=path.read_text().replace('        self.batch_histogram=Counter()', '        self.batch_histogram=Counter()\n        self._state_buffers={}')
s=s.replace('        states=np.zeros((padded_size,)+requests[0][0].shape,dtype=np.float32)', '''        states=self._state_buffers.get(padded_size)
        if states is None:
            states=np.zeros((padded_size,)+requests[0][0].shape,dtype=np.float32)
            self._state_buffers[padded_size]=states
        # device_get below completes inference before this buffer is reused.
        states[len(requests):].fill(0)''')
path.write_text(s)
print('Added action lookup and reusable inference buffers')
