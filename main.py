from __future__ import annotations

import os

import numpy as np

from utils.util import jst_today_str, jst_today_date
from utils.market import compute_market_score, compute_futures_risk
from utils.events import load_events, macro_caution_and_lines
from utils.state import get_weekly_new_count, increment_weekly_new_count, push_market_score
from utils.screener import run_screener
from utils.position import load_positions, analyze_positions
from utils.report import build_report
from utils.line import send_line

WORKER_URL = os.getenv("WORKER_URL")
WEEKLY_NEW_MAX = 3
MAX_DISPLAY = 5


def recommend_leverage(mkt_score: int, macro_caution: bool) -> float:
    # Conservative by design (v2.3). Keep fixed ladder.
    if mkt_score >= 75:
        lev = 1.1
    elif mkt_score >= 60:
        lev = 1.0
    elif mkt_score >= 50:
        lev = 1.0
    else:
        lev = 0.9
    if macro_caution:
        lev = min(lev, 1.1)  # output is guidance only; position sizing handled by user
    return float(lev)


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    market = compute_market_score()
    mkt_score = int(market.get("score", 50))

    futures_pct, risk_on = compute_futures_risk()

    # Events
    events = load_events()
    macro_caution, event_lines = macro_caution_and_lines(today_date, events)

    # Weekly new counter
    weekly_new = get_weekly_new_count()

    # Market score history -> delta3d
    delta3d = push_market_score(mkt_score)

    # Screener
    sres = run_screener(
        today_date=today_date,
        mkt_score=mkt_score,
        macro_caution=macro_caution,
        max_display=MAX_DISPLAY,
    )

    cands = sres["final"]
    rr_min = float(sres.get("rr_min", 2.2))

    # NO-TRADE logic (v2.3):
    #  - MarketScore<45 OR (delta3d<=-5 and MarketScore<55) => NO-TRADE
    #  - avgAdjEV<0.5 => NO-TRADE
    no_trade_reason = None
    if mkt_score < 45:
        no_trade_reason = "地合いNG（MarketScore<45）"
    elif delta3d <= -5 and mkt_score < 55:
        no_trade_reason = "地合い悪化（3日で-5以上）"
    elif float(sres.get("avg_adjev", 0.0)) < 0.5:
        no_trade_reason = "平均期待値<0.5"

    # Exception: if NO-TRADE but Tier0(A1-Strong) top 1 exists, allow show 1 slot (already included by screener).
    # Display decision is separated from trading permission. Permission remains NO.

    leverage = recommend_leverage(mkt_score, macro_caution)

    pos_df = load_positions("positions.csv")
    pos_text, _asset_est = analyze_positions(pos_df, mkt_score=mkt_score, macro_caution=macro_caution)

    report = build_report(
        today_str=today_str,
        market=market,
        delta3d=delta3d,
        futures_pct=futures_pct,
        risk_on=risk_on,
        macro_caution=macro_caution,
        weekly_new=weekly_new,
        weekly_new_max=WEEKLY_NEW_MAX,
        leverage=leverage,
        rr_min=rr_min,
        pos_text=pos_text,
        events_lines=event_lines,
        cands=cands,
        no_trade_reason=no_trade_reason,
    )

    print(report)
    send_line(report, WORKER_URL)

    # NOTE: Weekly new count is incremented manually by user when they actually enter.
    # This system is intentionally non-invasive.


if __name__ == "__main__":
    main()
