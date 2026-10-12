from pathlib import Path
import shutil
work=Path(__file__).resolve().parent
project=Path('/home/user/chess/alphazero_chess_v2')
for name in ('baseline','candidate'):
    shutil.copytree(project/'azchess',work/name/'azchess',dirs_exist_ok=True)
for name in ('verify_search.py','verify_single.py','verify_selfplay.py'):
    shutil.copy2(work.parent/'search_speedup_20261009'/name,work/name)
path=work/'candidate/azchess/batched.py'
s=path.read_text().replace('            move,node=agent.select_child(node);path.append(node);working.push(move);encoder.push()', '''            move,node=agent.select_child(node);path.append(node)
            # A terminal value belongs to this exact tree path and side to move.
            # No board/input reconstruction is needed when revisiting that leaf.
            if node.terminal_value is not None:break
            working.push(move);encoder.push()''')
s=s.replace('self._predict(self.agent.variables,jnp.asarray(states))', 'self._predict(self.agent.variables,states)')
path.write_text(s)
path=work/'candidate/azchess/mcts.py'
s=path.read_text().replace('                simulation_board.push(move)', '''                if node.terminal_value is not None:
                    break
                simulation_board.push(move)''')
# Cached terminal values are checked before board-dependent expansion.
path.write_text(s)
print('Prepared direct NumPy dispatch and cached terminal-path bypass')
