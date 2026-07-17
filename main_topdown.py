"""stockbotTOM — 新スクリーニング(topdown) entry point.

2026-07-13完全移行: GitHub Actionsの実行対象はこのファイル1本。
旧3システム(main_all.py / main.py / main_momentum.py と momentum/ swing2w/ mispricing/)は
コード温存・実行対象外(将来の比較・再併用のための保険として削除しない)。
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
from topdown import ledger

from mispricing.universe import load_universe
from mispricing.line_send import send_line

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
    print(f"[1/7] universe {len(uni)}銘柄 / dryrun={cfg.dryrun}")

    sentiment = compute_sentiment(cfg, dryrun=cfg.dryrun)
    print(f"[2/7] 地合い {sentiment['score']}/5{'(暫定)' if sentiment['provisional'] else ''} "
          f"{sentiment['stance']} VI代理={sentiment['vi_proxy'] if sentiment['vi_proxy'] is None else round(sentiment['vi_proxy'],1)} "
          f"semis={sentiment['semis_mode']}")

    tickers = uni["ticker"].tolist()
    ohlcv, meta = fetch_ohlcv(tickers, cfg.history_days, dryrun=cfg.dryrun)
    cov = meta.get("data_coverage", 0)
    print(f"[3/7] OHLCV {meta.get('data_ok','?')}/{meta.get('data_total','?')} ({cov*100:.0f}%)")
    meta = dict(meta)
    meta["data_warn"] = cov < cfg.data_coverage_min

    res = run_screen(uni, ohlcv, sentiment, cfg)
    st = res["stats"]
    print(f"[4/7] 母集団{st['eligible']} 点灯{st['trigger_count']} 急騰監視{st['spiked_watch']} "
          f"→ 本命{st['picked']}件 / 次点{len(res['watch'])}件")

    pos = load_positions_topdown(POSITIONS_PATH)
    pos_note = f"既存ポジ{len(pos)}件({POSITIONS_PATH})" if len(pos) else "既存ポジ: なし"
    position_alerts = check_held_positions(pos, uni, cfg, today=today)
    n_hit = sum(1 for a in position_alerts if a.get("hit"))
    print(f"[5/7] 保有銘柄チェック {len(position_alerts)}件(到達{n_hit}件)")

    plan_path = ledger.write_plan_log(cfg.outdir, today, res["picked"])
    ledger.ensure_result_template(cfg.outdir)
    ledger.append_reject_ledger(cfg.outdir, today, res["rejects"])
    print(f"[6/7] ログ出力 {plan_path}")

    events = load_week_events(cfg, today)
    text = build_text(today, meta, sentiment, res, position_alerts, pos_note, events, cfg)
    (outdir / f"topdown_report_{today}.txt").write_text(text, encoding="utf-8")

    png_path = str(outdir / f"topdown_report_table_{today}.png")
    images = []
    try:
        render_png(png_path, today, meta, sentiment, res, position_alerts, pos_note, events, cfg)
        images = [png_path]
    except Exception:
        print("[warn] PNG生成失敗(テキストのみ配信)")
        traceback.print_exc()

    result = send_line(text, image_paths=images,
                       image_caption=f"新スクリーニング {today} 本命{st['picked']}件 地合い{sentiment['score']}/5")
    print(f"[7/7] LINE result: {result}")

    require = str(os.getenv("REQUIRE_LINE_DELIVERY", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if require and not result.get("ok"):
        raise RuntimeError(f"LINE配信失敗: {result}")
    return {"ok": True, "picked": st["picked"], "line": result}


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # デッドマンズスイッチ: 例外時も可能な限りLINEに一報を入れてから落ちる(Actionsを赤にする)
        traceback.print_exc()
        try:
            send_line(f"⚠️stockbotTOM(新スクリーニング) エラーで停止: {type(e).__name__}: {e}\n"
                      f"詳細はGitHub Actionsのログを確認してください。")
        except Exception:
            pass
        raise
