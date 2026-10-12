from pathlib import Path
work=Path(__file__).resolve().parent
path=work/'candidate/azchess/batched.py'
s=path.read_text().replace('self._predict(self.agent.variables,jnp.zeros((size,8,8,BOARD_PLANES)))', 'self._predict(self.agent.variables,np.zeros((size,8,8,BOARD_PLANES),dtype=np.float32))')
path.write_text(s)
print('Warmup uses the same NumPy input path as actual inference')
