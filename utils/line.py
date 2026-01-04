from __future__ import annotations

from typing import Optional, List
import requests


def send_line(text: str, worker_url: Optional[str], chunk_size: int = 3800) -> None:
    """
    Cloudflare Worker 側が json={"text": "..."} を受け取る想定
    """
    if not worker_url:
        print(text)
        return

    chunks: List[str] = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        r = requests.post(worker_url, json={"text": ch}, timeout=25)
        print("[LINE RESULT]", r.status_code, str(r.text)[:200])