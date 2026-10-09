from pathlib import Path
import shutil
root=Path('/mnt/c/Users/User/Desktop/chess_v2')
path=root/'model_comparison.py'
shutil.copy2(path,path.with_name(path.name+'.pre-azrules-20261002'))
text=path.read_text(encoding='utf-8')
text=text.replace('NO_WINDOW =', 'LIVE_SAVED_MODELS = f"{PROJECT}/experiments/live-lr0001-20261002-run2/checkpoints/saved"\nNO_WINDOW =',1)
text=text.replace('("訪問回数版", RAW_SAVED_MODELS))','("訪問回数版", RAW_SAVED_MODELS), ("低学習率・自己対局", LIVE_SAVED_MODELS))')
text=text.replace('引き分けを請求できる局面では、請求するか指し続けるかをAIが選びます。','実際の3回反復・50手ルールで自動引き分け（AlphaZeroの終局条件）。')
text=text.replace('find {SAVED_MODELS} {RAW_SAVED_MODELS}', 'find {SAVED_MODELS} {RAW_SAVED_MODELS} {LIVE_SAVED_MODELS}')
text=text.replace("[/]train.py'", "[/]train.py|[/]run_live.py'")
path.write_text(text,encoding='utf-8')
# Both existing training controllers share the GPU; recognize the new runner.
for name in ['training_controller_v2_raw.py','training_controller_v2.py']:
    path=root/name
    if path.exists():
        text=path.read_text(encoding='utf-8')
        if '[/]run_live.py' not in text:
            shutil.copy2(path,path.with_name(path.name+'.pre-azrules-20261002'))
            text=text.replace("[/]arena.py'", "[/]arena.py|[/]run_live.py'")
            path.write_text(text,encoding='utf-8')
print('Updated comparison model list, rule description and GPU busy checks')
