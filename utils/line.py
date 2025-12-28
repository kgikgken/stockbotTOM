from __future__ import annotations

import os
import time
from typing import Optional

import requests

from utils.util import chunk_text


def send_line(text: str, worker_url: Optional[str] = None) -> None:
    """
    既存の「届く仕様」：
      POST WORKER_URL へ json={"text": "..."} を送る
    文字数が長い場合は分割送信（3800目安）
    """
    url = worker_url or os.getenv("WORKER_URL")
    if not url:
        print(text)
        return

    chunks = chunk_text(text, chunk_size=3800)

    for ch in chunks:
        ok = False
        last_err = None
        for _ in range(3):
            try:
                r = requests.post(url, json={"text": ch}, timeout=20)
                if 200 <= r.status_code < 300:
                    ok = True
                    break
                last_err = f"status={r.status_code} body={str(r.text)[:200]}"
            except Exception as e:
                last_err = str(e)
            time.sleep(0.6)

        if not ok:
            print("[LINE ERROR]", last_err)