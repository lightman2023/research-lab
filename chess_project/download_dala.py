"""Dala-900 の最新の公式ネットワークを models に保存する。"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

API_URL = "https://api.github.com/repos/hrschubert/dala-training/releases"
MODEL_PATTERN = re.compile(r"dala[-_]900.*\.pb\.gz$", re.IGNORECASE)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    model_dir = project_dir / "models"
    model_dir.mkdir(exist_ok=True)

    request = urllib.request.Request(
        API_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "chess-ai-setup"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        releases = json.load(response)

    candidates = [
        asset
        for release in releases
        for asset in release.get("assets", [])
        if MODEL_PATTERN.search(asset.get("name", ""))
    ]
    if not candidates:
        raise RuntimeError("公式リリースにDala-900の重みが見つかりませんでした。")

    asset = candidates[0]
    destination = model_dir / asset["name"]
    print(f"ダウンロード: {asset['name']}")
    urllib.request.urlretrieve(asset["browser_download_url"], destination)
    print(f"保存しました: {destination}")


if __name__ == "__main__":
    main()

