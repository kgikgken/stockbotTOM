"""stockbotTOM entry point.

Why `send_line` is resolved dynamically:
Some CI/runner layouts can accidentally import an unexpected `utils.line` module
or a stale cached file that does not expose `send_line`, causing an import-time
crash (ImportError: cannot import name 'send_line').

This file resolves the sender at runtime and keeps a safe fallback that
preserves the existing contract: `send_line(text: str) -> None`.
"""

from __future__ import annotations

import pandas as pd

from utils.util import jst_today_str, jst_today_date
from utils.market import market_score, futures_risk_on
from utils.events import build_event_section
from utils.state import (
    load_state,
    save_state,
    update_week,
    weekly_left,
    add_market_score,
    update_weekly_from_positions,
)
from utils.screener import run_screen
from utils.screen_logic import weekly_max_new, no_trade_conditions
from utils.position import load_positions, analyze_positions
from utils.report import build_report

from typing import Callable

def _resolve_send_line() -> Callable[[str], None]:
    """Resolve the LINE sender function.

    Prefer `utils.line.send_line` if available; otherwise, fall back to an
    internal implementation that matches the existing Worker contract.
    """
    # 1) Try to import our module and locate a compatible callable.
    try:
        import utils.line as _line  # type: ignore
    except Exception:
        _line = None  # type: ignore

    if _line is not None:
        for name in ("send_line", "send", "send_line_message"):
            fn = getattr(_line, name, None)
            if callable(fn):
                return fn  # type: ignore[return-value]

    # 2) Fallback: send via WORKER_URL if possible, otherwise print.
    def _fallback(text: str) -> None:
        import os
        import time

        worker_url = os.getenv("WORKER_URL")
        if not worker_url:
            print(text)
            return

        try:
            import requests  # local import: keep module import resilient
        except Exception:
            # If requests isn't available, do not fail the run.
            print(text)
            return

        chunk_size = 3800
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

        for ch in chunks:
            last_err = ""
            for attempt in range(3):
                try:
                    r = requests.post(worker_url, json={"text": ch}, timeout=20)
                    body = str(getattr(r, "text", ""))[:200]
                    print("[LINE RESULT]", getattr(r, "status_code", "?"), body)
                    if 200 <= int(getattr(r, "status_code", 0)) < 300:
                        last_err = ""
                        break
                    last_err = f"HTTP {getattr(r, 'status_code', '?')}: {body}"
                except Exception as e:
                    last_err = repr(e)
                time.sleep(0.8 * (2**attempt))
            if last_err:
                print("[LINE ERROR]", last_err)

    return _fallback


send_line = _resolve_send_line()

def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    state = load_state()
    update_week(state)

    # Load positions early: used both for reporting and for weekly new-count inference.
    pos_df = load_positions("positions.csv")
    try:
        pos_tickers = (
            pos_df.get("ticker", pd.Series(dtype=str)).astype(str).str.strip().tolist()
            if pos_df is not None and len(pos_df) else []
        )
    except Exception:
        pos_tickers = []
    prev_snapshot = state.get("positions_last", [])
    prev_set = set([str(x).strip() for x in prev_snapshot]) if isinstance(prev_snapshot, list) else set()
    new_tickers = sorted(list(set([str(x).strip() for x in pos_tickers if str(x).strip()]) - prev_set))
    update_weekly_from_positions(state, pos_tickers)

    mkt = market_score()
    mkt_score = int(mkt["score"])
    delta3 = add_market_score(state, today_str, mkt_score)

    risk_on, fut_chg = futures_risk_on()
    events_lines, macro_on = build_event_section(today_date)

    # weekly limit (based on inferred new positions)
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

    # New entry gate: include macro caution days as 'no new' while still showing a watchlist.
    no_trade = no_trade_conditions(mkt_score, delta3, macro_warn=macro_on)
    policy_lines = []
    if macro_on:
        policy_lines += [
            "新規は原則見送り（監視のみ）",
            "どうしても入るなら指値のみ",
            "ロットは50%以下",
            "リスク幅8%超は除外",
            "TP2は控えめ",
            "GUは寄り後再判定",
        ]
    else:
        policy_lines += [
            "新規は指値優先（現値INは条件達成銘柄のみ）",
            "リスク幅8%超は除外",
            "GUは寄り後再判定",
        ]
    # positions
    pos_text, _asset = analyze_positions(pos_df, mkt_score=mkt_score, macro_on=macro_on, new_tickers=new_tickers)

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
        saucers=meta.get("saucers"),
    )

    send_line(report)
    save_state(state)

if __name__ == "__main__":
    main()