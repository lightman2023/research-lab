from pathlib import Path
import shutil
import hashlib
import json
ROOT=Path('/home/user/chess/alphazero_chess_v2')
EXP=ROOT/'experiments/live-lr0001-temp1-20261002'
WORK=Path('/mnt/c/Users/User/Documents/ChatGPT/研究室/temperature1_gui_20261002')
WIN=Path('/mnt/c/Users/User/Desktop/chess_v2')
for directory,name in [(EXP,'README.md'),(WIN,'温度1学習README.md')]: shutil.copy2(WORK/'README.md',directory/name)
for name in ['setup.py','validate.py','test_gui_config.py','gui_config_test_results.json','test_windows_gui.py','windows_gui_test_results.json']:
    shutil.copy2(WORK/name,EXP/name)
shutil.copy2(WIN/'training_controller_v2_temp1.py',EXP/'code/training_controller_v2_temp1.py')
shutil.copy2(EXP/'carried_data_manifest.json',WORK/'carried_data_manifest.json')
for name in ['README.md']:
    path=ROOT/name
    text=path.read_text()
    if '## 温度1の追加学習GUI' not in text:
        text+='\n\n## 温度1の追加学習GUI\n\nWindowsの `v2温度1学習を起動.bat` を使用します。学習率0.0001一定・温度1の追加実験は `experiments/live-lr0001-temp1-20261002` に分離し、前回の最終モデルとAdamW状態、学習用80局を引き継ぎます。設定・操作・監視条件・検証結果は同実験のREADME.mdを参照してください。\n'
        path.write_text(text)
path=WIN/'README.md';text=path.read_text(encoding='utf-8')
text+='\n\n## 温度1の追加学習\n\n`v2温度1学習を起動.bat` から学習率0.0001・温度1の実験GUIを開けます。初期値は10サイクル、保存間隔1です。詳しくは `温度1学習README.md`。比較には既存の `v2モデル比較を起動.bat` の「低学習率・温度1」を使います。\n'
path.write_text(text,encoding='utf-8')
hashes={str(p.relative_to(EXP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in EXP.rglob('*') if p.is_file() and p.suffix in ['.py','.msgpack','.npz']}
(EXP/'setup_hashes.json').write_text(json.dumps(hashes,indent=2))
print('Saved setup, documentation and validation records')
