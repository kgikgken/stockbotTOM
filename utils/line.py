from __future__ import annotations

import requests


def send_line_message(text: str, worker_url: str | None) -> None:
    # Same delivered spec: POST json={"text": "..."} + chunking
    if not worker_url:
        print("[LINE] WORKER_URL not set; printed only.")
        return

    chunk_size = 3800
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    if not chunks:
        chunks = [""]

    for i, ch in enumerate(chunks, start=1):
        r = requests.post(worker_url, json={"text": ch}, timeout=20)
        print(f"[LINE RESULT] chunk={i}/{len(chunks)} status={r.status_code} body={str(r.text)[:200]}")