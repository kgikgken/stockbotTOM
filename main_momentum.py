"""stockbotTOM — モメンタム + 2週間スイング entry point.

★2026-07-12: 3システム統合(main_all.py)対応でパイプラインを関数化。
- run_pipeline(): 取得済みデータを受け取りレポートまで組み立てる(main_all.pyから共有呼び出し)
- main(): 単独実行用の従来エントリ(取得→run_pipeline→LINE送信)。挙動は従来と同一

ハード制約(ユーザー明示指定・裁量による例外なし):
- モメンタム: 実保有最大3銘柄・同一業種1銘柄まで・リスク固定0.5%(保有記録=positions.csv)
- 2週間スイング(swing2w): モメンタムとは★別枠★で実保有最大3銘柄・同一業種1銘柄まで・リスク固定0.5%
  (回転率二分のエンジンR/M、固定利確+時間ストップのハイブリッド出口。保有記録=positions_swing2w.csv)
レジーム防御モードは機械的ブロックではなく注意喚起のみ(モメンタム側のみに適用・ユーザー判断で変更)。
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
from momentum.position_check import check_held_positions, backfill_entry_scores
from momentum.report_text import build_text
from momentum.report_png import render_png
from momentum import ledger

from swing2w.config import load_config as load_config_swing2w
from swing2w.screen import run_screen as run_screen_swing2w
from swing2w.position_check import load_positions_swing2w, check_held_positions as check_held_swing2w

from mispricing.universe import load_universe, load_positions, open_risk_from_positions
from mispricing.line_send import send_line

JST = timezone(timedelta(hours=9))


def run_pipeline(uni, ohlcv: dict, meta: dict, cfg, today: str) -> dict:
    """モメンタム+swing2wパイプライン本体(データ取得とLINE送信を除く全て)。
    main_momentum.py単独実行とmain_all.py(3システム統合)の両方から呼ばれる。
    戻り値: {"text", "images", "caption", "picked", "picked_swing2w"}"""
    meta = dict(meta)  # 共有metaを汚さない
    meta["data_warn"] = meta["data_coverage"] < cfg.data_coverage_min
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- STEP1: レジームフィルター(警告のみ・ブロックしない) ----
    regime = compute_regime(cfg, today=today)
    print(f"[1/6] レジーム: {regime.get('mode')} attack={regime.get('attack')}")

    print(f"[2/6] OHLCV {meta['data_ok']}/{meta['data_total']} ({meta['data_coverage']*100:.0f}%)"
          + (f" (2巡目で{meta['recovered_2nd_pass']}件回収)" if meta.get("recovered_2nd_pass") else ""))
    if meta.get("fetch_failures"):
        ledger.append_fetch_failures(cfg.outdir, today, meta["fetch_failures"])
        print(f"       取得失敗 {len(meta['fetch_failures'])}件をmomentum_fetch_failures.csvに記録(指示⑬)")

    bench_logclose = None
    if regime.get("ok"):
        from momentum.data import fetch_regime_series
        bseries, _src = fetch_regime_series(cfg)
        if bseries is not None:
            bench_logclose = np.log(bseries)
    print(f"[3/6] ベンチマーク系列 {'取得済' if bench_logclose is not None else '未取得(相対強度なしで継続)'}")

    # ---- STEP2-3: 候補プール → 3状態分類 → アクション候補 ----
    res = run_screen(uni, ohlcv, bench_logclose, regime, cfg)
    print(f"[4/6] プール{res['stats']['pool_size']}銘柄 点灯{res['stats']['fired']}件 "
          f"→ 採用{res['stats']['picked']}件 state={res['stats']['state_count']}")
    gfc = res['stats'].get('gate_fail_counter', {})
    if gfc:
        print(f"[4.5/6] 状態A診断: trend_align=True但し非該当 {res['stats']['trend_align_true_n']}件 "
              f"— 落ちたゲート内訳(重複可): {dict(sorted(gfc.items(), key=lambda x: -x[1]))}")

    # ---- 既存ポジ注記 / entry_score自動転記(指示⑩) / 状態C・スコア劣化・TOB監視(指示③⑥⑦⑪) / ログ ----
    n_backfilled = backfill_entry_scores("positions.csv", res.get("eligible"), today, cfg.dryrun, cfg)
    if n_backfilled:
        print(f"[5.4/6] positions.csv の entry_score を{n_backfilled}件自動転記(指示⑩)")
    pos = load_positions("positions.csv")
    _, pos_note = open_risk_from_positions(pos, cfg.account_equity)
    position_alerts = check_held_positions(pos, uni, cfg, eligible=res.get("eligible"))
    if position_alerts:
        ledger.append_position_alerts(cfg.outdir, today, position_alerts)
    n_c = sum(1 for a in position_alerts if a.get("state_c"))
    n_sd = sum(1 for a in position_alerts if a.get("score_drop"))
    n_tob = sum(1 for a in position_alerts if a.get("tob_jump"))
    print(f"[5.5/6] 保有銘柄チェック {len(position_alerts)}件 (状態C{n_c}/スコア劣化{n_sd}/TOB急騰{n_tob})")

    ledger.append_reject_ledger(cfg.outdir, today, res["rejects"])
    plan_path = ledger.write_plan_log(cfg.outdir, today, res["picked"], regime, res["pool_stats"])
    ledger.ensure_result_template(cfg.outdir)
    print(f"[5/6] ログ出力 {plan_path}")

    # ---- swing2w(2週間スイング・別枠3銘柄・モメンタムと同じ取得済みohlcvを再利用) ----
    cfg2 = load_config_swing2w()
    res2 = run_screen_swing2w(uni, ohlcv, cfg2)
    st2 = res2["stats"]
    print(f"[5.6/6] swing2w 母集団{st2['universe_considered']}銘柄 "
          f"低回転{st2['low_turnover_n']}/高回転{st2['high_turnover_n']} "
          f"点灯R{st2['fired_r']}/M{st2['fired_m']} → 採用{st2['picked']}件")

    pos2 = load_positions_swing2w("positions_swing2w.csv")
    pos2_note = f"既存ポジ{len(pos2)}件(swing2w別枠)" if len(pos2) else "既存ポジ: なし(swing2w別枠)"
    position_alerts2 = check_held_swing2w(pos2, uni, cfg2, today=today)
    n_hit = sum(1 for a in position_alerts2 if a.get("hit"))
    print(f"[5.7/6] swing2w保有銘柄チェック {len(position_alerts2)}件(到達{n_hit}件)")

    # ---- report(モメンタム+swing2wを1本の統合レポートに) ----
    text = build_text(today, meta, regime, res, pos_note, cfg, position_alerts=position_alerts,
                      swing2w_res=res2, swing2w_alerts=position_alerts2,
                      swing2w_pos_note=pos2_note, swing2w_cfg=cfg2)
    (outdir / f"momentum_report_{today}.txt").write_text(text, encoding="utf-8")

    png_path = str(outdir / f"momentum_report_table_{today}.png")
    images: list[str] = []
    try:
        render_png(png_path, today, meta, regime, res, pos_note, cfg, position_alerts=position_alerts,
                  swing2w_res=res2, swing2w_alerts=position_alerts2,
                  swing2w_pos_note=pos2_note, swing2w_cfg=cfg2)
        images = [png_path]
    except Exception:
        traceback.print_exc()

    return {
        "text": text, "images": images,
        "caption": f"モメンタム{today} 候補{res['stats']['picked']}件 / "
                   f"2週間スイング候補{res2['stats']['picked']}件",
        "picked": res["stats"]["picked"], "picked_swing2w": res2["stats"]["picked"],
    }


def main() -> None:
    cfg = load_config()
    now_jst = datetime.now(JST)
    today = now_jst.strftime("%Y-%m-%d")

    uni = load_universe("universe_jpx.csv")
    if cfg.dryrun:
        uni = uni.head(60).copy()
    tickers = uni["ticker"].tolist()
    ohlcv, meta = fetch_ohlcv(tickers, cfg.history_days, dryrun=cfg.dryrun)

    out = run_pipeline(uni, ohlcv, meta, cfg, today)

    result = send_line(out["text"], image_paths=out["images"], image_caption=out["caption"])
    print("[6/6] LINE result:", {k: result.get(k) for k in
                                 ("ok", "text_ok", "image_ok", "backend", "status_code")})

    if os.getenv("REQUIRE_LINE_DELIVERY", "").strip().lower() in {"1", "true", "yes", "on"}:
        if not result.get("ok"):
            raise RuntimeError(f"LINE delivery failed: {result.get('reason', result)}")


if __name__ == "__main__":
    # ★指示⑮: デッドマンズスイッチ。正常終了時は通常の候補配信(0件でも実行できたことが分かる文面、
    # main()内で既に保証済み)。異常終了時もLINEへ必ず何か通知する(詳細はログのみ・LINEには種別のみ)。
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)  # 詳細なスタックトレースはGitHub Actionsログにのみ記録
        try:
            send_line(f"⚠️stockbotTOM(momentum) 本日のスクリーニングが実行エラーで停止しました\n"
                     f"エラー種別: {type(e).__name__}\n詳細はGitHub Actionsのログを確認してください")
        except Exception as ee:
            print(f"[WARN] LINE通知失敗: {ee}")
        raise
