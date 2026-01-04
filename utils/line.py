from __future__ import annotations

import time
import requests


def _chunks(text: str, chunk_size: int = 3800):
    if not text:
        return [""]
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def send_line_text(text: str, worker_url: str | None) -> None:
    """
    不達対策ど返し：
    - 3800文字分割
    - リトライ（指数バックオフ）
    - HTTP異常検知（2xx以外は失敗扱い）
    - timeout短め（20s）で固まらない
    """
    if not worker_url:
        print("[LINE] WORKER_URL not set. Print only.")
        return

    for idx, ch in enumerate(_chunks(text), 1):
        ok = False
        last_err = ""
        for attempt in range(1, 6):  # 5回
            try:
                r = requests.post(worker_url, json={"text": ch}, timeout=20)
                if 200 <= r.status_code < 300:
                    ok = True
                    print(f"[LINE] chunk {idx} OK status={r.status_code}")
                    break
                last_err = f"status={r.status_code} body={str(r.text)[:200]}"
            except Exception as e:
                last_err = repr(e)

            # backoff
            time.sleep(min(8.0, 0.7 * (2 ** (attempt - 1))))

        if not ok:
            # Actionsで検知したいので例外で落とす方が強いが、
            # ここは「送信失敗を必ずログに残す」目的で RuntimeError
            raise RuntimeError(f"[LINE] FAILED chunk {idx}: {last_err}")