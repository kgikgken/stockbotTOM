from __future__ import annotations

import os
import time
from typing import List

import requests


def send_line(text: str, worker_url: str | None = None) -> None:
    """Send text to LINE via Worker. Expects JSON {"text": "..."}.

    If WORKER_URL is missing, prints to stdout.
    """
    worker_url = worker_url or os.getenv("WORKER_URL")
    if not worker_url:
        print(text)
        return

    chunk_size = 3800  # LINE safety
    chunks: List[str] = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        try:
            r = requests.post(worker_url, json={"text": ch}, timeout=25)
            print("[LINE RESULT]", r.status_code, str(r.text)[:200])
            time.sleep(0.2)
        except Exception as e:
            print("[LINE ERROR]", type(e).__name__, str(e)[:200])
