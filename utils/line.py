from __future__ import annotations

import time
from typing import List

import requests


def _chunks(text: str, size: int) -> List[str]:
    if not text:
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]


def send_line(worker_url: str, text: str) -> None:
    """Cloudflare Worker へ投げる。届く仕様：POST json {"text": "..."}"""
    # LINE側制限・Worker制限の事故回避（安全マージン）
    chunk_size = 3800
    parts = _chunks(text, chunk_size)

    for part in parts:
        ok = False
        last_err = None
        for _ in range(3):
            try:
                r = requests.post(worker_url, json={"text": part}, timeout=20)
                if 200 <= r.status_code < 300:
                    ok = True
                    break
                last_err = f"status={r.status_code} body={str(r.text)[:200]}"
            except Exception as e:
                last_err = repr(e)
            time.sleep(0.8)
        if not ok:
            # 送信失敗でも止めない（Actionsは成功扱いにしたい/次回送れる方が重要）
            print("[LINE ERROR]", last_err)