# utils/line.py
from __future__ import annotations

import json
import requests
from typing import Optional


def line_notify(worker_url: str, message: str, timeout: int = 25) -> None:
    """
    Cloudflare Worker にPOSTしてLINEへ送る。
    Worker側は {"message": "..."} を受け取り、LINE Notify/APIへ配送する想定。
    """
    payload = {"message": message}

    r = requests.post(worker_url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=timeout)
    r.raise_for_status()


def line_notify_safe(worker_url: Optional[str], message: str) -> None:
    if not worker_url:
        print(message)
        return
    try:
        line_notify(worker_url, message)
    except Exception:
        # 送れない時も落とさない（ジョブ全体を殺さない）
        print(message)