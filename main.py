from __future__ import annotations

import os
import sys
from typing import Optional

from utils.screener import run_screening
from utils.report import build_report_text
from utils.line import send_line_message


def main() -> int:
    worker_url = os.getenv("WORKER_URL", "").strip()
    if not worker_url:
        print("ERROR: WORKER_URL is empty. Set WORKER_URL env var.", file=sys.stderr)
        return 2

    result = run_screening()

    text = build_report_text(result)

    ok = send_line_message(worker_url=worker_url, message=text)
    if not ok:
        print("ERROR: LINE send failed.", file=sys.stderr)
        return 1

    print("OK: LINE sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())