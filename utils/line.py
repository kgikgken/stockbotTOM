from __future__ import annotations

import time
from typing import List

import requests


def _chunk_text(text: str, chunk_size: int = 3800) -> List[str]:
    if not text:
        return [""]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def send_line_text(worker_url: str, text: str) -> None:
    """
    既に「届く」形を固定：
      POST WORKER_URL
      json={"text": "..."}
    """
    if not worker_url:
        print("[WARN] WORKER_URL is empty. Printing text only.")
        print(text)
        return

    chunks = _chunk_text(text, 3800)

    for idx, ch in enumerate(chunks, start=1):
        ok = False
        last_err = ""
        for _ in range(3):
            try:
                r = requests.post(worker_url, json={"text": ch}, timeout=20)
                if 200 <= r.status_code < 300:
                    ok = True
                    print(f"[LINE] chunk {idx}/{len(chunks)} OK {r.status_code}")
                    break
                last_err = f"status={r.status_code} body={str(r.text)[:200]}"
            except Exception as e:
                last_err = str(e)
            time.sleep(0.6)

        if not ok:
            print(f"[LINE] chunk {idx}/{len(chunks)} FAILED: {last_err}")