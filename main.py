from __future__ import annotations

import os
import numpy as np

from utils.util import jst_today_str, jst_today_date
from utils.market import enhance_market_score, calc_delta_market_score_3d, market_regime_multiplier
from utils.sector import top_sectors_5d
from utils.events import build_event_warnings, is_macro_danger
from utils.position import load_positions, analyze_positions, weekly_new_count
from utils.screener import run_swing_screening
from utils.report import build_report
from utils.line import send_line_text


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"

WORKER_URL = os.getenv("WORKER_URL")

# 週次制限（新規は週3まで）
WEEKLY_NEW_LIMIT = 3

# 資産フォールバック
ASSET_FALLBACK = 2_000_000.0


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    mkt_score = int(mkt.get("score", 50))
    delta3d = int(calc_delta_market_score_3d())

    # イベント
    event_lines = build_event_warnings(today_date, path=EVENTS_PATH)
    macro_danger = is_macro_danger(today_date, path=EVENTS_PATH)

    # ポジション
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt_score)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = ASSET_FALLBACK

    weekly_used = weekly_new_count(pos_df, today_date=today_date)
    weekly_remain = max(0, WEEKLY_NEW_LIMIT - weekly_used)

    # セクター
    sectors = top_sectors_5d(universe_path=UNIVERSE_PATH, top_n=5)

    # 地合い補正（AdjEV用）
    regime_mul = market_regime_multiplier(mkt_score=mkt_score, delta3d=delta3d, macro_danger=macro_danger)

    # スクリーニング（Swing）
    swing = run_swing_screening(
        today_date=today_date,
        universe_path=UNIVERSE_PATH,
        mkt_score=mkt_score,
        delta3d=delta3d,
        macro_danger=macro_danger,
        regime_mul=regime_mul,
        weekly_remain=weekly_remain,
    )

    # レポート生成
    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        delta3d=delta3d,
        weekly_used=weekly_used,
        weekly_limit=WEEKLY_NEW_LIMIT,
        macro_danger=macro_danger,
        sectors=sectors,
        event_lines=event_lines,
        swing=swing,
        pos_text=pos_text,
        total_asset=total_asset,
        regime_mul=regime_mul,
    )

    print(report)

    # LINE送信（不達ど返し）
    send_line_text(report, worker_url=WORKER_URL)


if __name__ == "__main__":
    main()