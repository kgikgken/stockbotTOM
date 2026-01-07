# ============================================
# utils/line.py
# LINE 通知送信（失敗耐性あり）
# ============================================

from __future__ import annotations

import os
import json
import requests
from typing import Optional


# --------------------------------------------
# 環境変数
# --------------------------------------------
LINE_WORKER_URL = os.getenv("WORKER_URL")  # Cloudflare Worker URL


# --------------------------------------------
# LINE 送信
# --------------------------------------------
def send_line_message(text: str) -> bool:
    """
    LINE にテキストを送信する
    Worker 経由。失敗しても例外は投げない
    """
    if not LINE_WORKER_URL:
        print("[WARN] WORKER_URL not set. Skip LINE send.")
        return False

    try:
        payload = {
            "message": text
        }

        res = requests.post(
            LINE_WORKER_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if res.status_code != 200:
            print(f"[WARN] LINE send failed: {res.status_code} {res.text}")
            return False

        return True

    except Exception as e:
        print(f"[ERROR] LINE send exception: {e}")
        return False