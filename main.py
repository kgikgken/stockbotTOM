from __future__ import annotations

from utils.market import build_market_context
from utils.screener import run_screener
from utils.report import build_report
from utils.line import send_line


def main() -> None:
    market = build_market_context()
    result = run_screener(market)
    text = build_report(market, result)
    print(text)
    send_line(text)


if __name__ == "__main__":
    main()