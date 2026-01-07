# ============================================
# utils/line.py
# Cloudflare Worker に投げて LINE 配信
# ============================================

from __future__ import annotations

import json
import time
from typing import Optional

import requests


def send_line_message(worker_url: str, message: str, max_retry: int = 3) -> bool:
    """
    WORKER_URL に対して { "message": "..."} を POST する。
    Worker 側で LINE 送信を実行する想定。
    """
    worker_url = (worker_url or "").strip()
    if not worker_url:
        return False

    payload = {"message": message}
    headers = {"Content-Type": "application/json"}

    last_err: Optional[str] = None
    for i in range(max_retry):
        try:
            r = requests.post(worker_url, data=json.dumps(payload), headers=headers, timeout=20)
            if 200 <= r.status_code < 300:
                return True
            last_err = f"status={r.status_code} body={r.text[:200]}"
        except Exception as e:
            last_err = str(e)

        time.sleep(1.2 * (i + 1))

    if last_err:
        print(f"[LINE] send failed: {last_err}")
    return False