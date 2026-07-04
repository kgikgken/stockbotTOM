"""stockbotTOM — v4.1歪みスクリーニング(パス1自動化) entry point.

旧トレンドフォロー(トレンドテンプレート/ソーサー)ロジックは全廃し、
v4.1ミスプライシング枠組みの「定量パス1」に全面置換。
utils/ パッケージへの依存なし(自己完結)。
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

from mispricing.config import load_config
from mispricing.universe import (load_universe, load_positions,
                                 open_risk_from_positions, load_week_events)
from mispricing.data import fetch_ohlcv, fetch_macro
from mispricing.macro import compute_macro
from mispricing.screen import run_screen
from mispricing.report_text import build_text
from mispricing.report_png import render_png
from mispricing import ledger
from mispricing.line_send import send_line

JST = timezone(timedelta(hours=9))


def main() -> None:
    cfg = load_config()
    today = datetime.now(JST).strftime("%Y-%m-%d")
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    uni = load_universe("universe_jpx.csv")
    if cfg.dryrun:
        uni = uni.head(60).copy()
    tickers = uni["ticker"].tolist()
    print(f"[1/6] universe {len(tickers)}銘柄 / dryrun={cfg.dryrun}")

    ohlcv, meta = fetch_ohlcv(tickers, cfg.history_days, dryrun=cfg.dryrun)
    meta["data_warn"] = meta["data_coverage"] < cfg.data_coverage_min
    print(f"[2/6] OHLCV {meta['data_ok']}/{meta['data_total']} "
          f"({meta['data_coverage']*100:.0f}%) warn={meta['data_warn']}")

    macro = compute_macro(fetch_macro(dryrun=cfg.dryrun), cfg)
    print(f"[3/6] 地合い {macro['score']}/5 VI={macro['vi']} lot={macro['lot_text']}")

    # ---- screen ----
    res = run_screen(uni, ohlcv, cfg, macro)
    print(f"[4/6] 検討{res['stats']['considered']} 棄却{res['stats']['rejected']} "
          f"仮点灯{res['stats']['picked']} 次点{len(res['runners'])}")

    # ---- STEP4 既存ポジ / STEP5 ログ ----
    pos = load_positions("positions.csv")
    _, pos_note = open_risk_from_positions(pos, cfg.account_equity)

    backfilled = ledger.backfill_reject_returns(cfg.outdir, ohlcv, cfg.reject_backfill_bdays)
    ledger.append_reject_ledger(cfg.outdir, today, res["rejects"])
    plan_path = ledger.write_plan_log(cfg.outdir, today, res["picked"], macro, cfg)
    ledger.ensure_result_template(cfg.outdir)
    print(f"[5/6] ログ出力 {plan_path} / 棄却台帳追記{len(res['rejects'])}件 / 追記{backfilled}件")

    # ---- report ----
    events = load_week_events("events.csv", datetime.now(JST).date())
    text = build_text(today, meta, macro, res, pos_note, events, backfilled, cfg)
    (outdir / f"report_{today}.txt").write_text(text, encoding="utf-8")

    png_path = str(outdir / f"report_table_{today}.png")
    try:
        render_png(png_path, today, meta, macro, res, pos_note, events, cfg)
        images = [png_path]
    except Exception:
        traceback.print_exc()
        images = []

    # ---- LINE ----
    result = send_line(text, image_paths=images,
                       image_caption=f"歪みスクリーニング {today} 仮点灯{res['stats']['picked']}件")
    print("[6/6] LINE result:", {k: result.get(k) for k in
                                 ("ok", "text_ok", "image_ok", "backend", "status_code")})

    if os.getenv("REQUIRE_LINE_DELIVERY", "").strip().lower() in {"1", "true", "yes", "on"}:
        if not result.get("ok"):
            raise RuntimeError(f"LINE delivery failed: {result.get('reason', result)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        try:
            send_line(f"stockbotTOM ERROR\n{type(e).__name__}: {e}\n(GitHub Actions log参照)")
        except Exception as ee:
            print(f"[WARN] LINE通知失敗: {ee}")
        raise
