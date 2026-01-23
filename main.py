# stockbotTOM main.py (Swing 1-7d) - v2.3 latest spec (base-structure fixed)
from __future__ import annotations

from utils.screener import run_screen
from utils.line import send_line


def main() -> int:
    report_text = run_screen()
    send_line(report_text)
    print(report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
