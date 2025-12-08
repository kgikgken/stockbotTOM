
import os
import requests

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

def main():
    # generate your real report here
    report_text = "test from GitHub Actions"
    send_line(report_text)

if __name__ == "__main__":
    main()
