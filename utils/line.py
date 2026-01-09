from __future__ import annotations

from typing import Optional, List

import requests


def send_line_text(text: str, worker_url: Optional[str], chunk_size: int = 3800) -> None:
    """
    ✅ LINE送信仕様（維持）
      POST WORKER_URL
      json={"text": "...chunk..."}
    """
    if not worker_url:
        print(text)
        return

    text = text or ""
    chunks: List[str] = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        r = requests.post(worker_url, json={"text": ch}, timeout=20)
        print("[LINE RESULT]", r.status_code, str(r.text)[:200])
