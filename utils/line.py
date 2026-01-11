from __future__ import annotations

import os
import requests

def send_line_text(text: str, worker_url: str | None = None) -> None:
    url = worker_url or os.getenv("WORKER_URL")
    if not url:
        print(text)
        return

    chunk_size = 3800
    for i in range(0, len(text), chunk_size):
        ch = text[i:i + chunk_size]
        r = requests.post(url, json={"text": ch}, timeout=20)
        print("[LINE]", r.status_code, str(r.text)[:200])
