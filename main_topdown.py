"""stockbotTOM — 新スクリーニング(topdown) entry point v2.0.

v2.0(2026-07-19)で追加された処理:
  - carryover: 候補の持ち越し(3営業日失効)・ゾーン到達の自動判定・反実仮想の記録
  - 出口の全面改訂に伴う保有監視の刷新(固定利確の撤廃)
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

from topdown.config import load_config
from topdown.data import fetch_ohlcv
from topdown.market import compute_sentiment
from topdown.screen import run_screen
from topdown.position_check import load_positions_topdown, check_held_positions, POSITIONS_PATH
from topdown.report_text import build_text, load_week_events
from topdown.report_png import render_png
from topdown import ledger, carryover
from topdown.universe import load_universe
from topdown.line_send import send_line

JST = timezone(timedelta(hours=9))


def main() -> dict:
    cfg = load_config()
    now = datetime.now(JST)
    today = now.strftime("%Y-%m-%d")
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    uni = load_universe("universe_jpx.csv")
    if cfg.dryrun:
        uni = uni.head(60)
    print(f"[1/8] universe {len(uni)}銘柄 / dryrun={cfg.dryrun}")

    sentiment = compute_sentiment(cfg, dryrun=cfg.dryrun)
    vi = sentiment.get("vi_proxy")
    print(f"[2/8] 地合い {sentiment['score']}/5{'(暫定)' if sentiment['provisional'] else ''} "
          f"{sentiment['stance']} VI代理={round(vi,1) if vi is not None else 'NA'} "
          f"semis={sentiment['semis_mode']}")

    tickers = uni["ticker"].tolist()
    ohlcv, meta = fetch_ohlcv(tickers, cfg.history_days, dryrun=cfg.dryrun)
    cov = meta.get("data_coverage", 0)
    print(f"[3/8] OHLCV {meta.get('data_ok','?')}/{meta.get('data_total','?')} ({cov*100:.0f}%)")
    meta = dict(meta)
    meta["data_warn"] = cov < cfg.data_coverage_min

    # --- 持ち越し候補の更新(ゾーン到達・失効の判定) ---
    pending = carryover.load_pending(carryover.PENDING_PATH)
    pending, pending_events = carryover.update_pending(pending, ohlcv, today, cfg)
    n_ev = len(pending_events)
    print(f"[4/8] 持ち越し候補 {len(pending)}件 / 本日の変化 {n_ev}件 "
          f"({', '.join(e['event'] for e in pending_events) if n_ev else '-'})")

    res = run_screen(uni, ohlcv, sentiment, cfg, today=today)
    st = res["stats"]
    print(f"[5/8] 母集団{st['eligible']} 点灯{st['trigger_count']} 急騰監視{st['spiked_watch']} "
          f"→ 本命{st['picked']}件 / 次点{len(res['watch'])}件")

    # --- 本日の候補を持ち越しリストへ追加 ---
    pending = carryover.add_new_candidates(pending, res["picked"], today,
                                           resolved_today=pending_events)
    pending = carryover.prune(pending, keep_days=90, today=today)
    carryover.save_pending(pending, carryover.PENDING_PATH)
    pending_summary = carryover.summarize(pending)

    pos = load_positions_topdown(POSITIONS_PATH)
    position_alerts = check_held_positions(pos, uni, cfg, today=today)
    n_hit = sum(1 for a in position_alerts if a.get("hit"))
    print(f"[6/8] 保有銘柄 {len(position_alerts)}件(到達{n_hit}件) / "
          f"ゾーン待ち{pending_summary['pending']}件")

    plan_path = ledger.write_plan_log(cfg.outdir, today, res["picked"])
    ledger.ensure_result_template(cfg.outdir)
    ledger.append_reject_ledger(cfg.outdir, today, res["rejects"])
    print(f"[7/8] ログ出力 {plan_path}")

    events = load_week_events(cfg, today)
    text = build_text(today, meta, sentiment, res, position_alerts,
                      pending_summary, pending_events, events, cfg)
    (outdir / f"topdown_report_{today}.txt").write_text(text, encoding="utf-8")

    png_path = str(outdir / f"topdown_report_table_{today}.png")
    images = []
    try:
        render_png(png_path, today, meta, sentiment, res, position_alerts,
                   pending_summary, pending_events, events, cfg)
        images = [png_path]
    except Exception:
        print("[warn] PNG生成失敗(テキストのみ配信)")
        traceback.print_exc()

    result = send_line(text, image_paths=images,
                       image_caption=f"新スクリーニング {today} 本命{st['picked']}件 "
                                     f"地合い{sentiment['score']}/5")
    print(f"[8/8] LINE result: {result}")

    require = str(os.getenv("REQUIRE_LINE_DELIVERY", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if require and not result.get("ok"):
        raise RuntimeError(f"LINE配信失敗: {result}")
    return {"ok": True, "picked": st["picked"], "line": result}


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        try:
            send_line(f"⚠️stockbotTOM(新スクリーニング) エラーで停止: {type(e).__name__}: {e}\n"
                      f"詳細はGitHub Actionsのログを確認してください。")
        except Exception:
            pass
        raise
