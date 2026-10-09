import json
import sys
import tkinter as tk
from pathlib import Path

sys.path.insert(0, 'C:/Users/User/Desktop/chess_v2')
from model_comparison_temp1 import Temperature1ComparisonApp

# Test actual WSL listing synchronously without a background Tk callback.
Temperature1ComparisonApp.refresh_models = lambda self: None
root = tk.Tk()
root.withdraw()
app = Temperature1ComparisonApp(root)
root.after = lambda delay, callback: callback()
app.refresh_models_worker()
assert app.model_a_var.get() == '低学習率・温度1 / cycle-000000.msgpack'
assert app.model_b_var.get() == '低学習率・温度1 / cycle-000010.msgpack'
assert app.games_var.get() == '24'
assert app.simulations_var.get() == '50'
assert app.max_plies_var.get() == '512'
assert not app.training_running()
for variable in (app.model_a_var, app.model_b_var):
    path = app.model_paths[variable.get()]
    unc = Path('//wsl.localhost/Ubuntu' + path)
    assert unc.is_file() and unc.stat().st_size > 0
result = dict(model_a=app.model_a_var.get(), model_b=app.model_b_var.get(),
              games=24, simulations=50, max_plies=512,
              models_exist=True, training_running=False, matches_started=False)
root.destroy()
Path(__file__).with_name('comparison_preset_validation.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=True))
