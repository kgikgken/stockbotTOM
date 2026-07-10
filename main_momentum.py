"""stockbotTOM — モメンタム・スクリーニング entry point(歪み系mispricing/とは完全独立).

「全振り」運用の実体: GitHub Actionsの実行対象を main.py からこのファイルに切り替えるだけで、
配信されるのはこのモメンタム・スクリーニングのみになる。main.py(歪み系)のコードは
削除せず温存(将来の比較・再併用のための保険)。

ハード制約(ユーザー明示指定・裁量による例外なし):
実保有最大3銘柄・同一業種1銘柄まで・リスク固定0.5%・レジーム防御モードで新規ゼロ。
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from momentum.config import load_config
from momentum.data import fetch_ohlcv
from momentum.regime import compute_regime
from momentum.screen import run_screen
from momentum.report_text import build_text
from momentum.report_png import render_png
from momentum import ledger

from mispricing.universe import load_universe, load_positions, open_risk_from_positions
from mispricing.line_send import send_line

JST = timezone(timedelta(hours=9))


def main() -> None:
    cfg = load_config()
    now_jst = datetime.now(JST)
    today = now_jst.strftime("%Y-%m-%d")
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- STEP1: レジームフィルター(絶対遵守・最優先) ----
    regime = compute_regime(cfg)
    print(f"[1/6] レジーム: {regime.get('mode')} attack={regime.get('attack')}")

    # ---- data ----
    uni = load_universe("universe_jpx.csv")
    if cfg.dryrun:
        uni = uni.head(60).copy()
    tickers = uni["ticker"].tolist()
    ohlcv, meta = fetch_ohlcv(tickers, cfg.history_days, dryrun=cfg.dryrun)
    meta["data_warn"] = meta["data_coverage"] < cfg.data_coverage_min
    print(f"[2/6] OHLCV {meta['data_ok']}/{meta['data_total']} ({meta['data_coverage']*100:.0f}%)")

    bench_logclose = None
    if regime.get("ok"):
        series, _ = None, None
        from momentum.data import fetch_regime_series
        bseries, _src = fetch_regime_series(cfg)
        if bseries is not None:
            bench_logclose = np.log(bseries)
    print(f"[3/6] ベンチマーク系列 {'取得済' if bench_logclose is not None else '未取得(相対強度なしで継続)'}")

    # ---- STEP2-3: 候補プール → 3状態分類 → アクション候補 ----
    res = run_screen(uni, ohlcv, bench_logclose, regime, cfg)
    print(f"[4/6] プール{res['stats']['pool_size']}銘柄 点灯{res['stats']['fired']}件 "
          f"→ 採用{res['stats']['picked']}件 state={res['stats']['state_count']}")

    # ---- 既存ポジ注記 / ログ ----
    pos = load_positions("positions.csv")
    _, pos_note = open_risk_from_positions(pos, cfg.account_equity)
    ledger.append_reject_ledger(cfg.outdir, today, res["rejects"])
    plan_path = ledger.write_plan_log(cfg.outdir, today, res["picked"], regime, res["pool_stats"])
    ledger.ensure_result_template(cfg.outdir)
    print(f"[5/6] ログ出力 {plan_path}")

    # ---- report ----
    text = build_text(today, meta, regime, res, pos_note, cfg)
    (outdir / f"momentum_report_{today}.txt").write_text(text, encoding="utf-8")

    png_path = str(outdir / f"momentum_report_table_{today}.png")
    try:
        render_png(png_path, today, meta, regime, res, pos_note, cfg)
        images = [png_path]
    except Exception:
        traceback.print_exc()
        images = []

    result = send_line(text, image_paths=images,
                       image_caption=f"モメンタム・スクリーニング {today} 候補{res['stats']['picked']}件")
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
            send_line(f"stockbotTOM(momentum) ERROR\n{type(e).__name__}: {e}\n(GitHub Actions log参照)")
        except Exception as ee:
            print(f"[WARN] LINE通知失敗: {ee}")
        raise
