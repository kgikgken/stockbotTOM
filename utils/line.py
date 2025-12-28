from __future__ import annotations

import requests
import os
import time


WORKER_URL = os.getenv("WORKER_URL")
CHUNK = 3800


def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return

    parts = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]

    for p in parts:
        try:
            r = requests.post(
                WORKER_URL,
                json={"text": p},
                timeout=20
            )
            time.sleep(0.5)
        except Exception as e:
            print("LINE ERROR:", e)