from __future__ import annotations

import os
from typing import Any, Dict, List

from utils.util import jst_today_str, jst_today
from utils.line import send_line
from utils.position import analyze_positions
from utils.report import build_report

# 1〜2ブロック目（中核ロジック）で作った想定
# - utils/market.py : build_market_context(today_date) -> dict
# - utils/screener.py : run_swing_screener(today_date, market_ctx) -> (swing_list, watch_list)
from utils.market import build_market_context
from utils.screener import run_swing_screener


def main() -> None:
    # 日付
    today_str = jst_today_str()
    today_date = jst_today()

    # 地合い（NO-TRADE判定までここで確定）
    market: Dict[str, Any] = build_market_context(today_date)

    # スクリーニング（仕様書：順張り / 追いかけ禁止 / 速度重視 / EV補正 / NO-TRADE強制）
    swing: List[Dict[str, Any]]
    watch: List[Dict[str, Any]]
    swing, watch = run_swing_screener(today_date, market)

    # ポジション表示
    pos_text = analyze_positions()

    # レポート
    report = build_report(
        date_str=today_str,
        market=market,
        swing=swing,
        watch=watch,
        positions=pos_text,
    )

    # 表示 & LINE
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()