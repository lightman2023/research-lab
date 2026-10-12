"""Cooperative MCTS: batch separate games while keeping each tree sequential."""
import numpy as np
import jax
import jax.numpy as jnp
from collections import Counter
from .game import legal_action_map
from .history import HistoryEncoder
from .mcts import Node

def search_requests(agent,board,rng):
    working=board.copy(stack=True);encoder=HistoryEncoder(working)
    root_ply=len(working.move_stack)
    def expand(node):
        terminal=encoder.terminal_value()
        if terminal is not None:return terminal
        priors,value=yield (encoder.encode(),legal_action_map(working))
        node.children={move:Node(prior) for move,prior in priors.items()}
        return value
    root=Node();yield from expand(root)
    if agent.root_noise and root.children:
        noise=rng.dirichlet([0.3]*len(root.children))
        for child,p in zip(root.children.values(),noise):child.prior=.75*child.prior+.25*float(p)
    for _ in range(agent.simulations):
        node=root;path=[node]
        while node.children:
            move,node=agent.select_child(node);path.append(node);working.push(move);encoder.push()
            if node.visit_count==0:break
        value=yield from expand(node)
        for visited in reversed(path):
            visited.visit_count+=1;visited.value_sum+=value;value=-value
        while len(working.move_stack)>root_ply:working.pop();encoder.pop()
    return root

class BatchedPredictor:
    def __init__(self,agent,batch_size):
        self.agent=agent;self.batch_size=batch_size;self.calls=0;self.positions=0
        self.batch_histogram=Counter()
        # Larger CUDA batches can otherwise select approximate TF32 kernels.
        # Keep full float32 products across batching configurations.
        def predict(variables,states):
            with jax.default_matmul_precision('highest'):
                return agent.model.apply(variables,states,train=False)
        self._predict=agent._predict if batch_size==1 else jax.jit(predict)
    def predict(self,requests):
        if not requests:return []
        if len(requests)>self.batch_size:raise ValueError('Too many inference requests')
        padded_size=min(self.batch_size,1<<(len(requests)-1).bit_length())
        states=np.zeros((padded_size,)+requests[0][0].shape,dtype=np.float32)
        for i,(state,_) in enumerate(requests):states[i]=state
        logits,values=jax.device_get(self._predict(self.agent.variables,jnp.asarray(states)))
        results=[]
        for i,(_,mapping) in enumerate(requests):
            indices=np.fromiter(mapping.keys(),dtype=np.int32)
            legal=logits[i,indices];legal-=legal.max();p=np.exp(legal);p/=p.sum()
            results.append(({mapping[index]:float(prob) for index,prob in zip(indices,p)},float(values[i])))
        self.calls+=1;self.positions+=len(requests)
        self.batch_histogram[padded_size]+=1
        return results

    def warmup(self):
        from .game import BOARD_PLANES
        sizes={self.batch_size};size=1
        while size<=self.batch_size:sizes.add(size);size*=2
        for size in sorted(sizes):
            output=self._predict(self.agent.variables,jnp.zeros((size,8,8,BOARD_PLANES)))
            jax.block_until_ready(output)
