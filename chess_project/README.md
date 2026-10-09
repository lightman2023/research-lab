# Dala-900 チェスAI

## GUIアプリで対局

エクスプローラーから `チェスAIを起動.bat` をダブルクリックします。
盤面では、動かしたい駒と移動先を順番にクリックしてください。

今後作成したLc0互換の `.pb.gz` モデルは `models` フォルダへ入れ、
アプリの「モデル一覧を更新」から選択できます。

WindowsのPowerShellで、上から順番に実行します。

## 1. フォルダへ移動

```powershell
cd "C:\Users\User\Desktop\chess"
```

## 2. Python環境を作成

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 3. Dala-900を取得

```powershell
python .\download_dala.py
python .\check_setup.py
```

## 4. lc0を用意

Leela Chess Zero公式リリースからWindows版をダウンロードして展開します。
展開先の `lc0.exe` の場所を確認してください。

## 5. 対局

次のパスは、実際の `lc0.exe` の場所に置き換えます。

```powershell
python .\play_dala.py --lc0 "C:\chess\lc0\lc0.exe" --color white
```

指し手は `e2e4`、`g1f3` の形式です。`quit`で終了します。
