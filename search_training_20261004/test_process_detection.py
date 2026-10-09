import re
from auto_search_training import BUSY_PATTERN

for name in ('parallel_selfplay.py','selfplay.py','train.py','arena.py','run_live.py','monitor_experiment.py','monitor_search_training.py','check_search_cycle.py'):
    assert re.search(BUSY_PATTERN,f'/home/user/chess/alphazero_chess_v2/.venv/bin/python -u {name} --checkpoint example')
    assert re.search(BUSY_PATTERN,f'python /home/user/chess/alphazero_chess_v2/{name} --experiment example')
assert not re.search(BUSY_PATTERN,f"pgrep -f '{BUSY_PATTERN}'")
assert not re.search(BUSY_PATTERN,'python auto_search_training.py')
print('PASS: relative/absolute processes detected; pgrep and autopilot excluded')
