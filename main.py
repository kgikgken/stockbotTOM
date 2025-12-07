import requests
import os

WORKER_URL = os.getenv("WORKER_URL")

def send_line(text: str):
    try:
        payload = {"text": text}
        r = requests.post(WORKER_URL, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print("LINE send error:", e)
        return False