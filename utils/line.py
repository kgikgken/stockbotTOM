from __future__ import annotations
import requests

def send_line_via_worker(worker_url: str, message: str, timeout: int = 20):
    r = requests.post(worker_url, json={"message": message}, timeout=timeout)
    return r.status_code == 200, r.text