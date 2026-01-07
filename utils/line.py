# ============================================
# utils/line.py
# LINE送信（Cloudflare Worker 経由）
# ============================================

import json
import urllib.request
import urllib.error


def send_line_message(worker_url: str, message: str) -> bool:
    """
    Cloudflare Worker に POST して LINE に送信する
    戻り値:
        True  : 送信成功
        False : 送信失敗（HTTPエラー含む）
    """

    if not worker_url:
        print("LINE ERROR: worker_url is empty")
        return False

    if not message:
        print("LINE ERROR: message is empty")
        return False

    payload = {
        "message": message
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        worker_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "stockbotTOM/1.0"
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            status = res.status
            body = res.read().decode("utf-8", errors="ignore")

        if status != 200:
            print(f"LINE ERROR: status={status} body={body}")
            return False

        return True

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"LINE HTTP ERROR: {e.code} body={body}")
        return False

    except urllib.error.URLError as e:
        print(f"LINE URL ERROR: {e}")
        return False

    except Exception as e:
        print(f"LINE UNKNOWN ERROR: {e}")
        return False