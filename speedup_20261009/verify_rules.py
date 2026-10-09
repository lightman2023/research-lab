from pathlib import Path
exec(Path(__file__).with_name('verify_and_benchmark.py').read_text().split('model,variables=')[0])
def oracle(base):
    class Oracle(base):
        def evaluate(self,board):
            mapping=game_old.legal_action_map(board)
            moves=list(mapping.values());weights=np.arange(1,len(moves)+1,dtype=float);weights/=weights.sum()
            return dict(zip(moves,weights)),0.123
    return Oracle(None,{},simulations=64,root_noise=True)
boards=[chess.Board(fen) for fen in [
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
    '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    '4k3/P7/8/8/8/8/7p/4K3 w - - 0 1',
    '7k/5K2/6R1/8/8/8/8/8 w - - 99 1',
    '8/8/8/8/8/8/4k3/7K w - - 100 1']]
for n in [7,8]:
    b=chess.Board()
    for move in (['g1f3','g8f6','f3g1','f6g8']*2)[:n]:b.push_uci(move)
    boards.append(b)
rng=np.random.default_rng(1109);b=chess.Board()
for ply in range(80):
    if b.is_game_over():break
    moves=list(b.legal_moves);b.push(moves[int(rng.integers(len(moves)))])
    if ply%8==0:boards.append(b.copy(stack=True))
for index,board in enumerate(boards):
    before=(board.fen(),tuple(board.move_stack))
    np.testing.assert_array_equal(game_old.encode_board(board),game_new.encode_board(board))
    assert game_old.legal_action_map(board)==game_new.legal_action_map(board)
    np.random.seed(900+index);a=oracle(old.MCTS).search(board)
    np.random.seed(900+index);b=oracle(new.MCTS).search(board)
    assert a.visit_count==b.visit_count and a.value_sum==b.value_sum
    assert [(m,n.visit_count,n.value_sum,n.prior) for m,n in a.children.items()]==[(m,n.visit_count,n.value_sum,n.prior) for m,n in b.children.items()]
    assert before==(board.fen(),tuple(board.move_stack))
(work/'rules_validation.json').write_text(json.dumps(dict(positions=len(boards),simulations=64,noise=True,
    includes=['castling','en_passant','promotion','50_move_boundary','threefold_history','insufficient_material'],identical=True)))
print('VERIFIED rules and complete history:',len(boards))
