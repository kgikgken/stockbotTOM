from __future__ import annotations

import os
import sys

from utils.screener import run_screening
from utils.report import build_report_text
from utils.line import send_line_message


def main() -> int:
    """
    stockbotTOM エントリーポイント
    - スクリーニング実行
    - レポート生成
    - LINE送信
    """

    worker_url = os.getenv("WORKER_URL", "").strip()
    if not worker_url:
        print("ERROR: WORKER_URL is not set", file=sys.stderr)
        return 2

    try:
        result = run_screening()
    except Exception as e:
        print(f"ERROR: screening failed: {e}", file=sys.stderr)
        return 1

    try:
        message = build_report_text(result)
    except Exception as e:
        print(f"ERROR: report build failed: {e}", file=sys.stderr)
        return 1

    ok = send_line_message(worker_url=worker_url, message=message)
    if not ok:
        print("ERROR: LINE send failed", file=sys.stderr)
        return 1

    print("OK: LINE sent successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())