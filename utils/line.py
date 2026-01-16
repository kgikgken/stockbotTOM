from __future__ import annotations

import os
import json
import urllib.request


def send_line(message: str) -> None:
    """LINE Notify 互換の簡易送信。

    環境変数：
      - LINE_WORKER_URL: Webhook エンドポイント
      - LINE_TOKEN: (任意) bearer token

    ※ 送信失敗してもプロセスを落とさない（CIでログ確認）
    """
    url = os.getenv("LINE_WORKER_URL")
    if not url:
        # ローカル/CIで未設定でも実行できるようにする
        return

    token = os.getenv("LINE_TOKEN", "").strip()
    payload = {"message": message}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            _ = resp.read()
    except Exception:
        # 送信はベストエフォート
        return
