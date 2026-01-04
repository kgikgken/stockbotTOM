# main.py
from __future__ import annotations

import os

from utils.util import jst_today_str
from utils.events import load_events
from utils.market import calc_market_state
from utils.sector import top_sectors_5d
from utils.screener import run_screening
from utils.report import build_line_report
from utils.line import line_notify


def main() -> None:
    universe_path = os.getenv("UNIVERSE_PATH", "universe_jpx.csv")
    positions_path = os.getenv("POSITIONS_PATH", "positions.csv")
    events_path = os.getenv("EVENTS_PATH", "events.csv")
    worker_url = os.getenv("WORKER_URL")  # Cloudflare Worker URL（LINE配送に使用）

    # 1) イベント（マクロ警戒）
    events = load_events(events_path)

    # 2) 地合い
    market = calc_market_state()

    # 3) セクター
    sectors = top_sectors_5d()

    # 4) スクリーニング（Swing専用）
    result = run_screening(
        universe_path=universe_path,
        positions_path=positions_path,
        events=events,
        market=market,
        sectors=sectors,
    )

    # 5) レポート生成
    msg = build_line_report(date_str=jst_today_str(), market=market, sectors=sectors, events=events, result=result)

    # 6) LINE送信
    if worker_url:
        line_notify(worker_url=worker_url, message=msg)
    else:
        # WORKER_URLが無いときは標準出力（ローカル確認用）
        print(msg)


if __name__ == "__main__":
    main()