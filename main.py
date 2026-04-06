"""stockbotTOM dual-screen entry point."""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

from utils.events import build_event_section
from utils.market import futures_risk_on, market_score
from utils.position import analyze_positions, load_positions
from utils.report import build_report
from utils.screen_logic import no_trade_conditions, weekly_max_new
from utils.screener import run_screen
from utils.state import add_market_score, load_state, save_state, update_week, update_weekly_from_positions, weekly_left
from utils.util import env_truthy, jst_today_date, jst_today_str


def _resolve_send_line() -> Callable[..., Dict]:
    try:
        import utils.line as _line  # type: ignore
    except Exception:
        _line = None  # type: ignore
    if _line is not None:
        for name in ("send_line", "send", "send_line_message"):
            fn = getattr(_line, name, None)
            if callable(fn):
                return fn

    def _fallback(text: str = "", *_args, **_kwargs) -> Dict:
        if text.strip():
            print(text)
        return {
            "ok": False,
            "text_ok": False,
            "image_ok": False,
            "reason": "stdout_fallback_only",
        }

    return _fallback


def _line_result_summary(result: Dict) -> Dict:
    try:
        from utils.line import summarize_line_result  # type: ignore

        return summarize_line_result(result)
    except Exception:
        keys = [
            "ok",
            "text_ok",
            "image_ok",
            "partial_image_ok",
            "reason",
            "text_mode",
            "image_mode",
            "text_status_code",
            "image_status_code",
        ]
        return {k: result.get(k) for k in keys}


send_line = _resolve_send_line()


def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    state = load_state()
    update_week(state)
    pos_df = load_positions("positions.csv")
    try:
        pos_tickers = (
            pos_df.get("ticker", pd.Series(dtype=str)).astype(str).str.strip().tolist()
            if pos_df is not None and len(pos_df)
            else []
        )
    except Exception:
        pos_tickers = []

    prev_snapshot = state.get("positions_last", [])
    prev_set = set(str(x).strip() for x in prev_snapshot) if isinstance(prev_snapshot, list) else set()
    new_tickers = sorted(set(str(x).strip() for x in pos_tickers if str(x).strip()) - prev_set)
    update_weekly_from_positions(state, pos_tickers)

    mkt = market_score()
    mkt_score = int(mkt.get("score", 55))
    delta3 = add_market_score(state, today_str, mkt_score)
    risk_on, fut_chg = futures_risk_on()
    events_lines, macro_on = build_event_section(today_date)
    weekly_used, weekly_max = weekly_left(state, max_new=weekly_max_new())

    leverage = 1.10 if mkt_score >= 60 else 1.00 if mkt_score >= 50 else 0.80
    if macro_on:
        leverage = min(leverage, 0.80)

    screen = run_screen(
        today_str=today_str,
        today_date=today_date,
        mkt_score=mkt_score,
        delta3=delta3,
        macro_on=macro_on,
        state=state,
    )
    trend_candidates = screen["trend_candidates"]
    leader_candidates = screen["leader_candidates"]
    meta = screen["meta"]

    no_trade = no_trade_conditions(int(meta.get("mkt_score_eff", mkt_score)), delta3, macro_warn=macro_on)
    if bool(meta.get("data_warn", False)) or bool(meta.get("breadth_force_no_trade", False)):
        no_trade = True

    policy_lines = [
        "dual lanes: trend + leaders",
        "trend lane keeps A1/A1-Strong/A2/B setup family",
        "leaders lane uses RS + 52W high + contraction + liquidity",
        "saucer lane removed",
        "risk width > 8% is excluded",
    ]
    if macro_on:
        policy_lines.insert(0, "macro caution: new entries should be smaller or watch-only")
    breadth_regime = str(meta.get("breadth_regime", ""))
    breadth_score = float(meta.get("breadth_score", 0.0))
    if breadth_regime:
        policy_lines.insert(0, f"breadth {breadth_regime} {breadth_score:.0f}")
    if bool(meta.get("data_warn", False)):
        ok = int(meta.get("data_ok", 0))
        total = int(meta.get("data_total", 0))
        cov = float(meta.get("data_coverage", 0.0))
        policy_lines.insert(0, f"data {ok}/{total} ({cov*100:.0f}%)")

    pos_text, positions_table = analyze_positions(pos_df, mkt_score=mkt_score, macro_on=macro_on, new_tickers=new_tickers)

    report = build_report(
        today_str=today_str,
        market=mkt,
        delta3=delta3,
        futures_chg=fut_chg,
        risk_on=risk_on,
        macro_on=macro_on,
        events_lines=events_lines,
        no_trade=no_trade,
        weekly_used=weekly_used,
        weekly_max=weekly_max,
        leverage=leverage,
        policy_lines=policy_lines,
        trend_candidates=trend_candidates,
        leader_candidates=leader_candidates,
        pos_text=pos_text,
        positions_df=positions_table,
    )

    image_paths = []
    if env_truthy("LINE_SEND_IMAGE", True):
        for key in ("summary_png", "trend_png", "leaders_png"):
            path = report.assets.get(key)
            if path and Path(path).exists():
                image_paths.append(path)

    require_delivery = env_truthy("REQUIRE_LINE_DELIVERY", False)
    require_images = env_truthy("REQUIRE_LINE_IMAGES", False)

    try:
        result = send_line(
            report.text,
            image_paths=image_paths,
            image_caption="",
            force_text=True,
            force_image=False,
        )
        print("LINE result:", _line_result_summary(result))

        if require_delivery and not bool(result.get("ok", False)):
            raise RuntimeError(f"LINE delivery failed: {_line_result_summary(result)}")
        if require_images and image_paths and not bool(result.get("image_ok", False)):
            raise RuntimeError(f"LINE image delivery failed: {_line_result_summary(result)}")
    except Exception:
        if require_delivery:
            raise
        traceback.print_exc()

    state["positions_last"] = sorted(set(str(x).strip() for x in pos_tickers if str(x).strip()))
    save_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        try:
            msg = f"stockbotTOM ERROR\n{type(e).__name__}: {e}\n\n(GitHub Actions log)"
            send_line(msg, force_text=True)
        except Exception as ee:
            print(f"[WARN] Failed to notify LINE: {ee}")
        raise
