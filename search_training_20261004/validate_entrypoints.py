import sys
import json
import tkinter as tk
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,'C:/Users/User/Desktop/chess_v2')
import model_comparison as base
from model_comparison_trained_search import TrainedSearchComparisonApp
with patch.object(base.ModelComparisonApp,'refresh_models',lambda self:None):
    root=tk.Tk();root.withdraw()
    app=TrainedSearchComparisonApp(root)
    root.after=lambda delay,fn:fn()
    app.refresh_models_worker()
    assert app.simulations_var.get()==app.simulations_b_var.get()=='200'
    assert app.max_plies_var.get()=='512'
    assert '探索学習A50 / cycle-000000.msgpack' in app.model_paths
    assert '探索学習B200 / cycle-000000.msgpack' in app.model_paths
    assert app.model_a_var.get()==app.model_b_var.get()==''
    assert 'まだ' in app.status_var.get()
    root.destroy()
result=dict(training_branches_listed=True,evaluation_both_200=True,
    incomplete_models_not_silently_replaced=True,full_evaluation_started=False)
Path(__file__).with_name('entrypoints_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
