from __future__ import annotations

import os
import requests


def send_line(message: str) -> None:
    """
    LINE Notify sender.
    If token is missing, prints only (never fail CI run).
    """
    token = os.getenv("LINE_NOTIFY_TOKEN") or os.getenv("LINE_TOKEN")
    if not token:
        print("[LINE] token missing; printing message only.")
        return

    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": "\n" + message}
    try:
        r = requests.post(url, headers=headers, data=data, timeout=20)
        if r.status_code != 200:
            print(f"[LINE] send failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[LINE] send exception: {e}")
