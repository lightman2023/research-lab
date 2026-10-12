import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,importlib,importlib.util,json
from pathlib import Path
import chess,numpy as np
work=Path(__file__).resolve().parent
project=Path('/home/user/chess/alphazero_chess_v2')
for label in ('baseline','candidate'):
    spec=importlib.util.spec_from_file_location(label,work/label/'azchess/__init__.py',submodule_search_locations=[str(work/label/'azchess')])
    module=importlib.util.module_from_spec(spec);sys.modules[label]=module;spec.loader.exec_module(module)
model,variables=importlib.import_module('candidate.checkpoint').load_variables(project/'experiments/search-budget-20261004/B200/checkpoints/saved/cycle-000050.msgpack')
boards=[chess.Board(),chess.Board('7k/5K2/6R1/8/8/8/8/8 w - - 99 1')]
nodes=0
def compare(a,b):
    global nodes
    assert list(a.children)==list(b.children)
    assert (a.visit_count,a.value_sum,a.prior)==(b.visit_count,b.value_sum,b.prior)
    nodes+=1
    for move in a.children:compare(a.children[move],b.children[move])
for board in boards:
    roots=[]
    for label in ('baseline','candidate'):
        agent=importlib.import_module(label+'.mcts').MCTS(model,variables,simulations=200,root_noise=True)
        np.random.seed(91009);roots.append(agent.search(board))
    compare(*roots)
report=dict(device='cpu',positions=2,simulations=200,exact_tree_nodes=nodes)
(work/'single_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
