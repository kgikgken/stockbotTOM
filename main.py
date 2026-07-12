"""stockbotTOM — v4.1歪みスクリーニング(パス1自動化) entry point.

旧トレンドフォロー(トレンドテンプレート/ソーサー)ロジックは全廃し、
v4.1ミスプライシング枠組みの「定量パス1」に全面置換。
utils/ パッケージへの依存なし(自己完結)。

★2026-07-12: 3システム統合(main_all.py)対応でパイプラインを関数化。
- run_pipeline(): 取得済みデータを受け取りレポートまで組み立てる(main_all.pyから共有呼び出し)
- main(): 単独実行用の従来エントリ(取得→run_pipeline→LINE送信)。挙動は従来と同一
- 保有ポジションは positions_mispricing.csv に分離(positions.csvはモメンタム専用。
  momentum側の状態C監視ロジックが歪みポジを誤判定しないようにするため)
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

POSITIONS_PATH = "positions_mispricing.csv"


def run_pipeline(uni_full, uni, ohlcv: dict, meta: dict, cfg, today: str, month: int) -> dict:
    """歪み×資金循環パイプライン本体(データ取得とLINE送信を除く全て)。
    main.py単独実行とmain_all.py(3システム統合)の両方から呼ばれる。
    戻り値: {"text", "images", "caption", "picked"}"""
    meta = dict(meta)  # 共有metaを汚さない(data_warnは各システムの基準で独立に判定)
    meta["data_warn"] = meta["data_coverage"] < cfg.data_coverage_min
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[2/8] OHLCV {meta['data_ok']}/{meta['data_total']} "
          f"({meta['data_coverage']*100:.0f}%) warn={meta['data_warn']}")

    macro = compute_macro(fetch_macro(dryrun=cfg.dryrun), cfg)
    print(f"[3/8] 地合い {macro['score']}/5 VI={macro['vi']} lot={macro['lot_text']}")

    flow = build_flow_map(uni, ohlcv, cfg)
    print(f"[4/8] 資金循環マップ ok={flow.get('ok')} レジーム={flow.get('regime')}")

    # ---- screen ----
    res = run_screen(uni, ohlcv, cfg, macro, flow, month)
    print(f"[5/8] 検討{res['stats']['considered']} 棄却{res['stats']['rejected']} "
          f"本命{res['stats']['picked']} 参考層{res['stats']['watch']} "
          f"engine={res['stats']['by_engine']}")

    # ---- STEP4 既存ポジ / STEP5 ログ ----
    pos = load_positions(POSITIONS_PATH)
    _, pos_note = open_risk_from_positions(pos, cfg.account_equity)
    pos_note = f"{pos_note}(歪み系・{POSITIONS_PATH})"
    positions_eval = check_positions(pos, uni_full, cfg, macro, flow, month)
    print(f"[6/8] 保有ポジション評価 {len(positions_eval)}件")

    backfilled = ledger.backfill_reject_returns(cfg.outdir, ohlcv, cfg.reject_backfill_bdays)
    ledger.append_reject_ledger(cfg.outdir, today, res["rejects"])
    plan_path = ledger.write_plan_log(cfg.outdir, today, res["picked"], macro, flow, cfg)
    ledger.ensure_result_template(cfg.outdir)
    print(f"[7/8] ログ出力 {plan_path} / 棄却台帳追記{len(res['rejects'])}件 / 追記{backfilled}件")

    # ---- report ----
    events = load_week_events("events.csv", datetime.now(JST).date())
    text = build_text(today, meta, macro, flow, res, pos_note, events, backfilled, cfg,
                      positions=positions_eval)
    (outdir / f"report_{today}.txt").write_text(text, encoding="utf-8")

    png_path = str(outdir / f"report_table_{today}.png")
    images: list[str] = []
    try:
        render_png(png_path, today, meta, macro, flow, res, pos_note, events, cfg,
                  positions=positions_eval)
        images = [png_path]
    except Exception:
        traceback.print_exc()

    return {
        "text": text, "images": images,
        "caption": f"歪み×資金循環スクリーニング {today} 本命{res['stats']['picked']}件",
        "picked": res["stats"]["picked"],
    }


def main() -> None:
    cfg = load_config()
    now_jst = datetime.now(JST)
    today = now_jst.strftime("%Y-%m-%d")

    # ---- data ----
    uni_full = load_universe("universe_jpx.csv")
    uni = uni_full.head(60).copy() if cfg.dryrun else uni_full
    tickers = uni["ticker"].tolist()
    print(f"[1/8] universe {len(tickers)}銘柄 / dryrun={cfg.dryrun}")

    ohlcv, meta = fetch_ohlcv(tickers, cfg.history_days, dryrun=cfg.dryrun)

    out = run_pipeline(uni_full, uni, ohlcv, meta, cfg, today, now_jst.month)

    # ---- LINE ----
    result = send_line(out["text"], image_paths=out["images"], image_caption=out["caption"])
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
