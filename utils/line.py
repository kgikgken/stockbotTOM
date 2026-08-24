from __future__ import annotations

import requests
from typing import Optional

def send_line(text: str, worker_url: Optional[str]) -> None:
    # 「届く」仕様：json={"text": "..."} のPOST
    if not worker_url:
        print(text)
        return

    chunk_size = 3800  # LINE文字制限回避
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        try:
            r = requests.post(worker_url, json={"text": ch}, timeout=20)
            print("[LINE RESULT]", r.status_code, str(r.text)[:200])
        except Exception as e:
            # 落とさない（届かないのが最悪）
            print("[LINE ERROR]", repr(e))
