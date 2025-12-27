from __future__ import annotations

import os
from typing import List

import requests

def send_line(text: str, worker_url: str | None = None) -> None:
    url = worker_url or os.getenv("WORKER_URL")
    if not url:
        print(text)
        return

    chunk_size = 3800
    chunks: List[str] = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        r = requests.post(url, json={"text": ch}, timeout=20)
        print("[LINE]", r.status_code)
