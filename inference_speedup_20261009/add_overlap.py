from pathlib import Path
work=Path(__file__).resolve().parent
path=work/'candidate/azchess/batched.py'
s=path.read_text().replace('yield (encoder.encode(),legal_action_map(working,moves))', '''yield (encoder.encode(),lambda: legal_action_map(working,moves))''')
s=s.replace('        logits,values=jax.device_get(self._predict(self.agent.variables,states))', '''        pending_output=self._predict(self.agent.variables,states)
        # Each game's generator is paused until its response is delivered, so
        # its board remains unchanged while these deferred mappings are built.
        prepared=[]
        for _,mapping in requests:
            if callable(mapping):mapping=mapping()
            prepared.append((mapping,np.fromiter(mapping.keys(),dtype=np.int32)))
        logits,values=jax.device_get(pending_output)''')
s=s.replace('        for i,(_,mapping) in enumerate(requests):\n            indices=np.fromiter(mapping.keys(),dtype=np.int32)', '        for i,(mapping,indices) in enumerate(prepared):')
path.write_text(s)
print('Deferred legal-action mapping overlaps GPU inference')
