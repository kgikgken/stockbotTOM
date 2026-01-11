from __future__ import annotations

import os
import traceback

from utils.util import jst_today_str, jst_today_date
from utils.market import enhance_market_score
from utils.state import load_state, update_state_after_run
from utils.events import load_events, macro_warning_block
from utils.position import load_positions, analyze_positions
from utils.screener import run_screening
from utils.report import build_report
from utils.line import send_line_text

UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    state = load_state(path="state.json")

    mkt = enhance_market_score()
    mkt_score = int(mkt.get("score", 50))

    # stateからΔ3d（無ければ0）
    delta3d = float(state.get("delta3d", 0.0) or 0.0)

    # 先物Risk-ON：先物+1% OR (MarketScore>=65 & Δ3d>=+3)
    fut_risk_on = bool(mkt.get("futures_risk_on", False) or (mkt_score >= 65 and delta3d >= 3.0))

    events = load_events(EVENTS_PATH)
    macro_on, macro_lines = macro_warning_block(events, today_date)

    pos_df = load_positions(POSITIONS_PATH)
    pos_text = analyze_positions(pos_df, mkt_score=mkt_score, macro_on=macro_on)

    screen = run_screening(
        universe_path=UNIVERSE_PATH,
        today_date=today_date,
        mkt=mkt,
        macro_on=macro_on,
        futures_risk_on=fut_risk_on,
        state=state,
    )

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        delta3d=delta3d,
        macro_on=macro_on,
        macro_lines=macro_lines,
        futures_risk_on=fut_risk_on,
        state=state,
        screen=screen,
        pos_text=pos_text,
    )

    print(report)
    send_line_text(report, worker_url=WORKER_URL)

    update_state_after_run(path="state.json", state=state, mkt_score=mkt_score)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        msg = "stockbotTOM 実行エラー\n\n" + err[-3500:]
        print(msg)
        send_line_text(msg, worker_url=WORKER_URL)
        raise
