from __future__ import annotations

import os

from utils.util import jst_today_str, jst_today_date
from utils.market import market_score, futures_risk_on
from utils.events import build_event_section
from utils.state import load_state, save_state, update_week, weekly_left, add_market_score
from utils.screener import run_screen
from utils.screen_logic import weekly_max_new, no_trade_conditions
from utils.position import load_positions, analyze_positions
from utils.report import build_report
from utils.line import send_line

def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    state = load_state()
    update_week(state)

    mkt = market_score()
    mkt_score = int(mkt["score"])
    delta3 = add_market_score(state, today_str, mkt_score)

    risk_on, fut_chg = futures_risk_on()
    events_lines, macro_on = build_event_section(today_date)

    # weekly limit
    used, wmax = weekly_left(state, max_new=weekly_max_new())

    # leverage suggestion (simple)
    leverage = 1.1 if mkt_score >= 50 else 0.9
    if macro_on:
        leverage = min(leverage, 1.1)

    # screening
    cands, meta, _ohlc_map = run_screen(
        today_str=today_str,
        today_date=today_date,
        mkt_score=mkt_score,
        delta3=delta3,
        macro_on=macro_on,
        state=state,
    )

    no_trade = no_trade_conditions(mkt_score, delta3)
    policy_lines = []
    if macro_on:
        policy_lines += ["新規は指値のみ", "ロットは50%以下", "TP2は控えめ", "GUは寄り後再判定"]
    else:
        policy_lines += ["新規は指値のみ（現値IN禁止）"]

    # positions
    pos_df = load_positions("positions.csv")
    pos_text, _asset = analyze_positions(pos_df, mkt_score=mkt_score, macro_on=macro_on)

    report = build_report(
        today_str=today_str,
        market=mkt,
        delta3=delta3,
        futures_chg=fut_chg,
        risk_on=risk_on,
        macro_on=macro_on,
        events_lines=events_lines,
        no_trade=no_trade,
        weekly_used=used,
        weekly_max=wmax,
        leverage=leverage,
        policy_lines=policy_lines,
        cands=cands,
        pos_text=pos_text,
    )

    send_line(report)
    save_state(state)

if __name__ == "__main__":
    main()
