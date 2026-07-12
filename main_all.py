"""stockbotTOM — 3システム統合entry point(2026-07-12・歪み系の実行復帰).

日次実行(GitHub Actions)はこのファイル1本。以下を1回のデータ取得で全て実行する:
  ①モメンタム(最大3・1業種1・シャンデリア出口)+ ②2週間スイング(別枠3・固定利確+時間ストップ)
     → 統合レポート1本目としてLINE配信
  ③歪み×資金循環(業種内リバーサル2-5日・positions_mispricing.csv別管理)
     → レポート2本目としてLINE配信

設計上のポイント:
- 実データ取得は1回だけ(momentum.data.fetch_ohlcv = 2巡目リトライ+取得失敗記録つき)。
  3システムが同一のデータスナップショットを見るため、システム間で「同じ日なのに違うデータ」
  という不整合が起きない。DRYRUNのみ各系統の合成データを別々に取得(テストフィクスチャ温存のため)。
- 片方のシステムが実行エラーでも、もう片方は配信される(ブロック単位の隔離)。
  エラーがあった場合は最後にまとめてraiseし、GitHub Actionsを失敗(赤)にする。
- 保有記録は3ファイル分離: positions.csv(モメンタム)/ positions_swing2w.csv / positions_mispricing.csv。
  各システムが自分の保有だけを自分のロジックで監視する(相互の誤判定を防ぐ)。
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone, timedelta

import main as mispricing_entry
import main_momentum as momentum_entry

from mispricing.config import load_config as load_config_mispricing
from mispricing.data import fetch_ohlcv as fetch_ohlcv_mispricing
from mispricing.universe import load_universe
from mispricing.line_send import send_line
from momentum.config import load_config as load_config_momentum
from momentum.data import fetch_ohlcv as fetch_ohlcv_momentum

JST = timezone(timedelta(hours=9))


def main() -> None:
    cfg_mom = load_config_momentum()
    cfg_mis = load_config_mispricing()
    now_jst = datetime.now(JST)
    today = now_jst.strftime("%Y-%m-%d")

    # ---- データ取得(実データは1回だけ・3システム共有) ----
    uni_full = load_universe("universe_jpx.csv")
    uni = uni_full.head(60).copy() if cfg_mom.dryrun else uni_full
    tickers = uni["ticker"].tolist()
    print(f"[0] universe {len(tickers)}銘柄 / dryrun={cfg_mom.dryrun}")

    if cfg_mom.dryrun:
        # DRYRUNは系統ごとの合成データを別々に生成(各系統のテストフィクスチャを温存)
        ohlcv_mom, meta_mom = fetch_ohlcv_momentum(tickers, cfg_mom.history_days, dryrun=True)
        ohlcv_mis, meta_mis = fetch_ohlcv_mispricing(tickers, cfg_mis.history_days, dryrun=True)
    else:
        hist = max(cfg_mom.history_days, cfg_mis.history_days)
        ohlcv, meta = fetch_ohlcv_momentum(tickers, hist, dryrun=False)  # 2巡目リトライ+失敗記録つき
        ohlcv_mom, meta_mom = ohlcv, meta
        ohlcv_mis, meta_mis = ohlcv, meta
        print(f"[0] OHLCV共有取得 {meta['data_ok']}/{meta['data_total']} "
              f"({meta['data_coverage']*100:.0f}%) — 3システムで同一スナップショットを使用")

    errors: list[str] = []
    line_results: list[dict] = []

    # ---- ①② モメンタム + 2週間スイング(統合レポート・LINE 1本目) ----
    try:
        out1 = momentum_entry.run_pipeline(uni, ohlcv_mom, meta_mom, cfg_mom, today)
        r1 = send_line(out1["text"], image_paths=out1["images"], image_caption=out1["caption"])
        line_results.append(r1)
        print("[LINE 1/2 momentum+swing2w]:", {k: r1.get(k) for k in
                                               ("ok", "text_ok", "image_ok", "backend", "status_code")})
    except Exception:
        traceback.print_exc()
        errors.append("momentum+swing2w")

    # ---- ③ 歪み×資金循環(LINE 2本目) ----
    try:
        out2 = mispricing_entry.run_pipeline(uni_full, uni, ohlcv_mis, meta_mis, cfg_mis,
                                             today, now_jst.month)
        r2 = send_line(out2["text"], image_paths=out2["images"], image_caption=out2["caption"])
        line_results.append(r2)
        print("[LINE 2/2 歪み×資金循環]:", {k: r2.get(k) for k in
                                          ("ok", "text_ok", "image_ok", "backend", "status_code")})
    except Exception:
        traceback.print_exc()
        errors.append("歪み×資金循環")

    if os.getenv("REQUIRE_LINE_DELIVERY", "").strip().lower() in {"1", "true", "yes", "on"}:
        for r in line_results:
            if not r.get("ok"):
                raise RuntimeError(f"LINE delivery failed: {r.get('reason', r)}")

    if errors:
        raise RuntimeError(f"実行エラーのシステム: {' / '.join(errors)}(もう一方は配信済み)")


if __name__ == "__main__":
    # デッドマンズスイッチ: 異常終了時もLINEへ必ず通知(詳細はGitHub Actionsログのみ)
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        try:
            send_line(f"⚠️stockbotTOM(統合) 本日のスクリーニングでエラーが発生しました\n"
                     f"エラー種別: {type(e).__name__}: {e}\n詳細はGitHub Actionsのログを確認してください")
        except Exception as ee:
            print(f"[WARN] LINE通知失敗: {ee}")
        raise
