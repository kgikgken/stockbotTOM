"""stockbotTOM — v5.0 歪み×資金循環スクリーニング(パス1自動化) entry point.

旧トレンドフォロー(v2.0系)・v4.1タイプ体系は全廃し、v5.0エンジン体系
(A=業種内リバーサル / S=逆流戻り売り / B疑い=PEAD)へ再編。
資金循環マップ(STEP1.5)と保有ポジション日次評価を搭載。utils/依存なし。
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
from mispricing.flowmap import build_flow_map
from mispricing.screen import run_screen
from mispricing.position_check import check_positions
from mispricing.report_text import build_text
from mispricing.report_png import render_png
from mispricing import ledger
from mispricing.line_send import send_line

JST = timezone(timedelta(hours=9))


def main() -> None:
    cfg = load_config()
    now_jst = datetime.now(JST)
    today = now_jst.strftime("%Y-%m-%d")
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    uni_full = load_universe("universe_jpx.csv")
    uni = uni_full.head(60).copy() if cfg.dryrun else uni_full
    tickers = uni["ticker"].tolist()
    print(f"[1/8] universe {len(tickers)}銘柄 / dryrun={cfg.dryrun}")

    ohlcv, meta = fetch_ohlcv(tickers, cfg.history_days, dryrun=cfg.dryrun)
    meta["data_warn"] = meta["data_coverage"] < cfg.data_coverage_min
    print(f"[2/8] OHLCV {meta['data_ok']}/{meta['data_total']} "
          f"({meta['data_coverage']*100:.0f}%) warn={meta['data_warn']}")

    macro = compute_macro(fetch_macro(dryrun=cfg.dryrun), cfg)
    print(f"[3/8] 地合い {macro['score']}/5 VI={macro['vi']} lot={macro['lot_text']}")

    flow = build_flow_map(uni, ohlcv, cfg)
    print(f"[4/8] 資金循環マップ ok={flow.get('ok')} レジーム={flow.get('regime')}")

    # ---- screen ----
    res = run_screen(uni, ohlcv, cfg, macro, flow, now_jst.month)
    print(f"[5/8] 検討{res['stats']['considered']} 棄却{res['stats']['rejected']} "
          f"本命{res['stats']['picked']} 参考層{res['stats']['watch']} "
          f"engine={res['stats']['by_engine']}")

    # ---- STEP4 既存ポジ / 保有評価 / STEP5 ログ ----
    pos = load_positions("positions.csv")
    _, pos_note = open_risk_from_positions(pos, cfg.account_equity)
    positions_eval = check_positions(pos, uni_full, cfg, macro, flow, now_jst.month)
    print(f"[6/8] 保有ポジション評価 {len(positions_eval)}件")

    backfilled = ledger.backfill_reject_returns(cfg.outdir, ohlcv, cfg.reject_backfill_bdays)
    ledger.append_reject_ledger(cfg.outdir, today, res["rejects"])
    plan_path = ledger.write_plan_log(cfg.outdir, today, res["picked"], macro, flow, cfg)
    ledger.ensure_result_template(cfg.outdir)
    print(f"[7/8] ログ出力 {plan_path} / 棄却台帳追記{len(res['rejects'])}件 / 追記{backfilled}件")

    # ---- report ----
    events = load_week_events("events.csv", now_jst.date())
    text = build_text(today, meta, macro, flow, res, pos_note, events, backfilled, cfg,
                      positions=positions_eval)
    (outdir / f"report_{today}.txt").write_text(text, encoding="utf-8")

    png_path = str(outdir / f"report_table_{today}.png")
    try:
        render_png(png_path, today, meta, macro, flow, res, pos_note, events, cfg,
                  positions=positions_eval)
        images = [png_path]
    except Exception:
        traceback.print_exc()
        images = []

    # ---- LINE ----
    result = send_line(text, image_paths=images,
                       image_caption=f"歪み×資金循環スクリーニング {today} 本命{res['stats']['picked']}件")
    print("[8/8] LINE result:", {k: result.get(k) for k in
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
