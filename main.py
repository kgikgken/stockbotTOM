from __future__ import annotations

import os
import sys

from utils.util import jst_today_str, jst_today_date
from utils.market import enhance_market_score, recommend_leverage, calc_max_position
from utils.sector import top_sectors_5d
from utils.events import build_event_warnings
from utils.position import load_positions, analyze_positions
from utils.screener import run_swing_screening
from utils.report import build_report
from utils.line import send_line


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    # 市況
    mkt = enhance_market_score()
    mkt_score = int(mkt.get("score", 50))
    delta3d = float(mkt.get("delta3d", 0.0))

    # ポジション
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt_score)
    if total_asset <= 0:
        total_asset = 2_000_000.0

    # レバ/建玉
    lev, lev_comment = recommend_leverage(mkt_score, delta3d=delta3d)
    max_pos = calc_max_position(total_asset, lev)

    # セクター / イベント
    sectors = top_sectors_5d(top_n=5)
    events = build_event_warnings(today_date, events_path=EVENTS_PATH)

    # スクリーニング
    swing_result = run_swing_screening(
        today_date=today_date,
        universe_path=UNIVERSE_PATH,
        mkt=mkt,
        positions_df=pos_df,
    )

    # レポート
    report = build_report(
        today_str=today_str,
        mkt=mkt,
        lev=lev,
        lev_comment=lev_comment,
        max_position=max_pos,
        sectors=sectors,
        events=events,
        swing=swing_result,
        pos_text=pos_text,
        total_asset=total_asset,
    )

    print(report)

    # LINE送信（届く方式：json={"text": "..."} を分割送信）
    if WORKER_URL:
        send_line(WORKER_URL, report)
    else:
        # WORKER_URL無いときは標準出力のみ
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)