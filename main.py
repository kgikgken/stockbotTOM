"""stockbotTOM entry point."""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Callable, Dict

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


def _resolve_send_line() -> Callable[..., Dict]:
    """Resolve LINE sender with a conservative fallback."""
    try:
        import utils.line as _line  # type: ignore
    except Exception:
        _line = None  # type: ignore

    if _line is not None:
        for name in ("send_line", "send", "send_line_message"):
            fn = getattr(_line, name, None)
            if callable(fn):
                return fn  # type: ignore[return-value]

    def _fallback(text: str = "", *_args, **_kwargs) -> Dict:
        worker_url = os.getenv("WORKER_URL", "").strip()
        if not text.strip():
            return {
                "ok": True,
                "text_ok": True,
                "image_ok": False,
                "skipped": True,
                "reason": "no_content",
                "backend": "fallback",
                "status_code": 0,
            }
        if not worker_url:
            print(text)
            return {
                "ok": True,
                "text_ok": True,
                "image_ok": False,
                "skipped": True,
                "reason": "printed_stdout",
                "backend": "stdout",
                "status_code": 0,
            }
        try:
            import requests
            r = requests.post(worker_url, json={"text": text}, timeout=20)
            ok = 200 <= int(getattr(r, "status_code", 0) or 0) < 300
            return {
                "ok": ok,
                "text_ok": ok,
                "image_ok": False,
                "skipped": False,
                "reason": "text_only" if ok else "text_failed",
                "backend": "fallback-worker",
                "status_code": int(getattr(r, "status_code", 0) or 0),
                "body": (getattr(r, "text", "") or "")[:300],
            }
        except Exception as e:
            print(text)
            return {
                "ok": False,
                "text_ok": False,
                "image_ok": False,
                "skipped": False,
                "reason": str(e),
                "backend": "fallback-worker",
                "status_code": 0,
            }

    return _fallback


send_line = _resolve_send_line()


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


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
    mkt_score = int(mkt["score"])
    delta3 = add_market_score(state, today_str, mkt_score)

    risk_on, fut_chg = futures_risk_on()
    events_lines, macro_on = build_event_section(today_date)

    used, wmax = weekly_left(state, max_new=weekly_max_new())
    leverage = 1.1 if mkt_score >= 50 else 0.9
    if macro_on:
        leverage = min(leverage, 1.1)

    cands, meta, _ohlc_map = run_screen(
        today_str=today_str,
        today_date=today_date,
        mkt_score=mkt_score,
        delta3=delta3,
        macro_on=macro_on,
        state=state,
    )

    no_trade = no_trade_conditions(mkt_score, delta3, macro_warn=macro_on)
    data_warn = bool(meta.get("data_warn", False))
    breadth_warn = bool(meta.get("breadth_warn", False))
    breadth_score = int(meta.get("breadth_score", 50) or 50)
    if data_warn or (breadth_warn and mkt_score < 55):
        no_trade = True

    policy_lines = [
        "新規は指値優先（現値INは条件達成銘柄のみ）",
        "リスク幅8%超は除外",
        "GUは寄り後再判定",
    ]
    if macro_on:
        policy_lines = [
            "新規は原則見送り（監視のみ）",
            "どうしても入るなら指値のみ",
            "ロットは50%以下",
            "リスク幅8%超は除外",
            "TP2は控えめ",
            "GUは寄り後再判定",
        ]
    if data_warn:
        ok = int(meta.get("data_ok", 0))
        tot = int(meta.get("data_total", 0))
        cov = float(meta.get("data_coverage", 0.0))
        cov_min = float(meta.get("data_coverage_min", 0.0))
        policy_lines.insert(0, f"DATA:{ok}/{tot} ({cov*100:.0f}%) < {cov_min*100:.0f}%")
    if breadth_warn:
        regime = str(meta.get("breadth_regime", "weak"))
        a20 = float(meta.get("breadth_above20", 0.0) or 0.0)
        a50 = float(meta.get("breadth_above50", 0.0) or 0.0)
        policy_lines.insert(0, f"BREADTH:{breadth_score} ({regime}) / >20MA {a20:.0f}% / >50MA {a50:.0f}%")

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

    out_dir = Path(os.getenv("REPORT_OUTDIR", "out"))
    image_paths = []
    if _env_truthy("LINE_SEND_IMAGE", True):
        candidates = [
            out_dir / f"report_table_{today_str}.png",
            out_dir / f"report_table_{today_str}_d.png",
            out_dir / f"report_table_{today_str}_w.png",
            out_dir / f"report_table_{today_str}_m.png",
        ]
        image_paths = [str(p) for p in candidates if p.exists()]

    require_delivery = _env_truthy("REQUIRE_LINE_DELIVERY", False)
    require_images = _env_truthy("REQUIRE_LINE_IMAGES", False)

    try:
        result: Dict = {}
        if image_paths:
            result = send_line(
                "",
                image_paths=image_paths,
                image_caption="",
                force_image=True,
            )
        else:
            result = send_line(report, force_text=True)

        print("LINE result:", result)

        if require_delivery:
            if image_paths:
                if not bool(result.get("ok", False)):
                    reason = str(result.get("reason", "LINE delivery failed"))
                    raise RuntimeError(f"LINE delivery failed: {reason}")
            else:
                if not bool(result.get("text_ok", False)):
                    reason = str(result.get("reason", "LINE text delivery failed"))
                    raise RuntimeError(f"LINE delivery failed (text): {reason}")

        if require_images and image_paths and not bool(result.get("image_ok", False)):
            reason = str(result.get("reason", "one or more images failed"))
            raise RuntimeError(f"LINE delivery failed (one or more images): {reason}")
    except Exception:
        if require_delivery:
            raise
        traceback.print_exc()

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
