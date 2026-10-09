import importlib.util
import json
from pathlib import Path
import tkinter as tk
path=Path('C:/Users/User/Desktop/chess_v2/training_controller_v2_temp1.py')
spec=importlib.util.spec_from_file_location('temp1_gui',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=tk.Tk();root.withdraw()
app=module.Controller(root);root.update_idletasks()
assert app.run_cycles_var.get()=='10'
assert app.save_interval_var.get()=='1'
assert app.simulations_var.get()=='50'
assert not app.train_button.instate(['disabled'])
assert app.stop_button.instate(['disabled'])
result=dict(windows_tk_initialized=True,title=root.title(),cycles_per_run=app.run_cycles_var.get(),save_interval=app.save_interval_var.get(),buttons_initialized=True,learning_started=False)
root.destroy()
Path(__file__).with_name('windows_gui_test_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True))
