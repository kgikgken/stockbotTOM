"""stockbotTOM entry point.

Why `send_line` is resolved dynamically:
Some CI/runner layouts can accidentally import an unexpected `utils.line` module
or a stale cached file that does not expose `send_line`, causing an import-time
crash (ImportError: cannot import name 'send_line').

This file resolves the sender at runtime and keeps a safe fallback that
preserves the existing contract: `send_line(text: str) -> None`.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

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

from typing import Callable, Optional

def _resolve_send_line() -> Callable[..., None]:
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
    def _fallback(text: str, *_args, **_kwargs) -> None:
        import os
        import time

        # When running in image-only mode, `main()` may call send_line("", image_path=...).
        # In the fallback (text-only) sender, avoid pushing empty messages.
        if not str(text or "").strip():
            return {
                "ok": True,
                "skipped": True,
                "status_code": 0,
                "text": "",
                "image_ok": False,
            }

        worker_url = os.getenv("WORKER_URL")
        if not worker_url:
            print(text)
            return {
                "ok": True,
                "status_code": 0,
                "text": str(text),
                "image_ok": False,
            }

        try:
            import requests  # local import: keep module import resilient
        except Exception:
            # If requests isn't available, do not fail the run.
            print(text)
            return {
                "ok": True,
                "status_code": 0,
                "text": str(text),
                "image_ok": False,
            }

        chunk_size = 3800
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

        for ch in chunks:
            last_err = ""
            last_status = 0
            last_body = ""
            for attempt in range(3):
                try:
                    r = requests.post(worker_url, json={"text": ch}, timeout=20)
                    last_status = int(getattr(r, "status_code", 0) or 0)
                    last_body = str(getattr(r, "text", ""))[:200]
                    print("[LINE RESULT]", last_status, last_body)
                    if 200 <= last_status < 300:
                        last_err = ""
                        break
                    last_err = f"HTTP {last_status}: {last_body}"
                except Exception as e:
                    last_err = repr(e)
                time.sleep(0.8 * (2**attempt))
            if last_err:
                print("[LINE ERROR]", last_err)

        ok = not bool(last_err)
        return {
            "ok": ok,
            "status_code": last_status,
            "text": last_body,
            "image_ok": False,
        }

    return _fallback


send_line = _resolve_send_line()


def _env_truthy(name: str, default: bool = False) -> bool:
    """Parse a boolean-like environment variable.

    Accepts common truthy/falsy strings:
    - truthy: 1, true, yes, y, on
    - falsy : 0, false, no, n, off
    (case-insensitive)
    """

    raw = os.getenv(name)
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default

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
    # Data coverage gating (yfinance instability)
    data_warn = bool(meta.get("data_warn", False))
    if data_warn:
        no_trade = True
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
    if data_warn:
        ok = int(meta.get("data_ok", 0))
        tot = int(meta.get("data_total", 0))
        cov = float(meta.get("data_coverage", 0.0))
        cov_min = float(meta.get("data_coverage_min", 0.0))
        policy_lines.insert(0, f"DATA:{ok}/{tot} ({cov*100:.0f}%) < {cov_min*100:.0f}%")

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

    # Prefer sending the table image(s) to LINE as well (worker must support multipart upload).
    # We may generate multiple pages:
    #   - report_table_YYYY-MM-DD.png   : 注文（狙える + ポジション）
    #   - report_table_YYYY-MM-DD_d.png : ソーサー（日足）
    #   - report_table_YYYY-MM-DD_w.png : ソーサー（週足）
    #   - report_table_YYYY-MM-DD_m.png : ソーサー（月足）(該当時のみ)
    image_paths: list[str] = []
    out_dir = os.getenv("REPORT_OUTDIR", "out")
    if _env_truthy("LINE_SEND_IMAGE", True):
        candidates = [
            Path(out_dir) / f"report_table_{today_str}.png",
            Path(out_dir) / f"report_table_{today_str}_d.png",
            Path(out_dir) / f"report_table_{today_str}_w.png",
            Path(out_dir) / f"report_table_{today_str}_m.png",
        ]
        image_paths = [str(p) for p in candidates if p.exists()]

    require_delivery = _env_truthy("REQUIRE_LINE_DELIVERY", False)
    # If True, fail the run when at least one image page could not be delivered.
    # Default is False: it's usually better to still deliver the text report
    # (or a partial set of images) than to hard-fail the whole workflow.
    require_images = _env_truthy("REQUIRE_LINE_IMAGES", False)

    if send_line:
        try:
            result: dict = {}
            if image_paths:
                # Images-only delivery in LINE.
                # (The summary/title is embedded in the PNG, so we intentionally send NO text.)
                first = image_paths[0]
                result = send_line(
                    "",  # IMPORTANT: images-only
                    image_path=first,
                    image_caption="",  # image only (no caption text)
                    image_key=os.path.basename(first),
                )
                others: list[dict] = []
                for p in image_paths[1:]:
                    others.append(
                        send_line(
                            "",
                            image_path=p,
                            image_caption="",  # image only (no caption text)
                            image_key=os.path.basename(p),
                        )
                    )

                # Aggregate status across pages
                image_ok_all = bool(result.get("image_ok"))
                for r in others:
                    if r.get("image_ok") is False:
                        image_ok_all = False
                result["image_ok_all"] = image_ok_all

                # Fail-safe: if any image failed, also send the text report so LINE still arrives.
                if not image_ok_all:
                    print("[WARN] One or more images failed to send. Falling back to text report.")
                    try:
                        send_line(report)
                    except Exception as ee:
                        print(f"[WARN] Failed to send fallback text: {ee}")
            else:
                # No images → fallback to text
                result = send_line(report)

            print("LINE result:", result)

            if require_delivery:
                if image_paths:
                    # Require that at least the *first* message (report + first image)
                    # was delivered. Additional image pages are best-effort unless
                    # REQUIRE_LINE_IMAGES=1.
                    if not result.get("ok", False):
                        raise RuntimeError("LINE delivery failed")
                else:
                    if not result.get("text_ok", False):
                        raise RuntimeError("LINE delivery failed (text)")

            if require_images and image_paths:
                if not result.get("image_ok_all", False):
                    raise RuntimeError("LINE delivery failed (one or more images)")
        except Exception as e:
            print("LINE delivery error:", e)
            if require_delivery:
                raise

    save_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail-safe: still notify LINE so you notice breakages quickly.
        tb = traceback.format_exc()
        print(tb)
        try:
            msg = f"stockbotTOM ERROR\n{type(e).__name__}: {e}\n\n(GitHub Actions log)"
            # send text even if LINE_SEND_TEXT is false
            send_line(msg, force_text=True)
        except Exception as ee:
            print(f"[WARN] Failed to notify LINE: {ee}")
        raise
