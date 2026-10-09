import json
import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,'C:/Users/User/Desktop/chess_v2')
import model_comparison as base
from model_comparison_search import SearchComparisonApp

with patch.object(base.ModelComparisonApp,'refresh_models',lambda self:None):
    root=tk.Tk();root.withdraw()
    app=SearchComparisonApp(root)
    root.after=lambda delay,callback:callback()
    app.refresh_models_worker()
    label='低学習率・温度1 / cycle-000010.msgpack'
    assert app.model_a_var.get()==app.model_b_var.get()==label
    assert app.simulations_var.get()=='50' and app.simulations_b_var.get()=='200'
    assert app.max_plies_var.get()=='512'
    assert not app.training_running()
    with patch.object(base.threading,'Thread') as thread, patch.object(app,'training_running',return_value=False):
        app.start_match()
        arguments=thread.call_args.kwargs['args']
        assert arguments[0]==arguments[1]
        assert arguments[2:6]==(24,50,200,512)
        assert app.sims_b_spin.instate(['disabled'])
    class Process:
        stdout=[]
        def wait(self):return 0
    with patch.object(base.subprocess,'Popen',return_value=Process()) as process:
        app.match_worker(*arguments)
        command=process.call_args.args[0][-1]
        assert '--simulations-a 50 --simulations-b 200' in command
    app.set_running(False)
    app.simulations_b_var.set('50')
    with patch.object(base.messagebox,'showwarning') as warning, patch.object(base.threading,'Thread') as thread:
        app.start_match()
        assert warning.called and not thread.called
    root.destroy()
result=dict(gui_initializes=True,same_model_allowed_with_different_search=True,
    simulations_a=50,simulations_b=200,max_plies=512,command_forwards_both=True,
    identical_conditions_rejected=True,full_evaluation_started=False)
Path(__file__).with_name('search_comparison_gui_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
