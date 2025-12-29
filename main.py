import os
import sys
from typing import Optional

from utils.util import jst_today_str, jst_today_date
from utils.market import enhance_market_score
from utils.position import load_positions, analyze_positions
from utils.screener import run_screening
from utils.report import build_report
from utils.line import send_line


UNIVERSE_PATH = os.getenv("UNIVERSE_PATH", "universe_jpx.csv")
POSITIONS_PATH = os.getenv("POSITIONS_PATH", "positions.csv")
EVENTS_PATH = os.getenv("EVENTS_PATH", "events.csv")
WORKER_URL = os.getenv("WORKER_URL")


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=int(mkt.get("score", 50)))
    if total_asset <= 0:
        total_asset = 2_000_000.0

    result = run_screening(
        universe_path=UNIVERSE_PATH,
        events_path=EVENTS_PATH,
        today_date=today_date,
        market=mkt,
        total_asset=total_asset,
    )

    text = build_report(
        today_str=today_str,
        today_date=today_date,
        market=mkt,
        screening=result,
        pos_text=pos_text,
        total_asset=total_asset,
    )

    # Stdout (Actionsログ用)
    print(text)

    # LINE送信（届く仕様：json={"text": "..."}）
    if WORKER_URL:
        send_line(WORKER_URL, text)
    else:
        # WORKER_URL 未設定なら stdout のみ
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # GitHub Actions で原因が見えるようにだけはして落とす
        print("[FATAL]", repr(e), file=sys.stderr)
        raise