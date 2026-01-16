from __future__ import annotations

import os
from typing import List

import requests

WORKER_URL = os.getenv("WORKER_URL")

def send_line(text: str) -> None:
    """
    Cloudflare Worker へ POST json={"text": "..."} を送る（既存仕様維持）
    """
    if not WORKER_URL:
        print(text)
        return

    chunk_size = 3800
    chunks: List[str] = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        ok = False
        last_err = ""
        for attempt in range(3):
            try:
                r = requests.post(WORKER_URL, json={"text": ch}, timeout=20)
                print("[LINE RESULT]", r.status_code, str(r.text)[:200])
                ok = True
                break
            except Exception as e:
                last_err = repr(e)
        if not ok:
            # Never fail the whole run because LINE is temporarily unavailable.
            print("[LINE ERROR]", last_err)
