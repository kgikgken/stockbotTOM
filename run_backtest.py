"""バックテスト実行スクリプト(実データ用・GitHub Actions等ネット接続がある環境で実行すること)。

設計:
- mispricing/data.py の実データ取得ロジックを再利用し、多年分のヒストリカルOHLCVを取得する。
- 各銘柄のDataFrameにsector/name情報をattrsとして埋め込む(backtest/run_*.pyがそこから読む)。
- momentum・swing2wそれぞれのウォークフォワード・バックテストを実行し、
  トレードログ(CSV)とサマリー統計をout_backtest/に出力する。

★既知の限界(必ず結果と一緒に確認すること):
- yfinanceは生存者バイアスを含む(現在上場中の銘柄のみ。上場廃止銘柄は欠落)。
  この限界により、結果は実際より楽観的に出る傾向がある。
- 手数料・スプレッド・市場インパクトは未考慮(約定は日足High/Low/Open/Closeのみで判定)。
- 対TOPIX相対強度・業種内相対リターンは、その時点でyfinanceから取得できた銘柄群のみで
  計算される(当時実際に存在した全銘柄群とは一致しない可能性がある)。
- 実行時間: 全銘柄(約3786)×数年分は非常に重い処理になる。初回はUNIVERSE_LIMIT環境変数で
  銘柄数を絞ってのテスト実行を推奨する。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from functools import partial

import pandas as pd

from mispricing.universe import load_universe
from mispricing.data import fetch_ohlcv as fetch_ohlcv_real

from momentum.config import load_config as load_momentum_config
from swing2w.config import load_config as load_swing2w_config
from backtest.engine import run_walk_forward
from backtest.run_momentum import momentum_signal_fn, momentum_exit_fn
from backtest.run_swing2w import swing2w_signal_fn, swing2w_exit_fn
from backtest.metrics import summarize, trades_to_dataframe


def main():
    t0 = time.time()
    outdir = Path("out_backtest")
    outdir.mkdir(parents=True, exist_ok=True)

    history_days = int(os.getenv("BACKTEST_HISTORY_DAYS", "1500"))  # 既定約6年
    universe_limit = int(os.getenv("UNIVERSE_LIMIT", "0"))  # 0=制限なし(全銘柄・非常に重い)

    uni = load_universe("universe_jpx.csv")
    if universe_limit > 0:
        uni = uni.head(universe_limit).copy()
        print(f"[注意] UNIVERSE_LIMIT={universe_limit}で銘柄数を絞ったテスト実行")

    tickers = uni["ticker"].tolist()
    print(f"[1/5] 銘柄数: {len(tickers)} / 取得年数目安: {history_days}営業日分")

    ohlcv, meta = fetch_ohlcv_real(tickers, history_days, dryrun=False)
    print(f"[2/5] OHLCV取得 {meta['data_ok']}/{meta['data_total']} ({meta['data_coverage']*100:.0f}%)")

    # sector/nameをattrsに埋め込む(backtest/run_*.pyがdf.attrsから読む設計のため必須)
    uni_idx = uni.set_index("ticker")
    for tkr, df in ohlcv.items():
        if tkr in uni_idx.index:
            df.attrs["sector"] = uni_idx.loc[tkr, "sector"]
            df.attrs["name"] = uni_idx.loc[tkr, "name"]

    # TOPIX代理(1306)を取得(momentumの対TOPIX相対強度用)
    bench_ohlcv, _ = fetch_ohlcv_real(["1306.T"], history_days, dryrun=False)
    bench_df = bench_ohlcv.get("1306.T")
    if bench_df is None:
        print("[警告] TOPIX代理(1306)の取得に失敗。対TOPIX相対強度・セクター強度は無効化されます")

    all_dates = sorted(set().union(*[set(df.index) for df in ohlcv.values()]))
    trading_dates = pd.DatetimeIndex(all_dates)
    print(f"[3/5] 対象期間: {trading_dates[0].date()} 〜 {trading_dates[-1].date()}({len(trading_dates)}営業日)")

    # ---- momentum ----
    cfg_m = load_momentum_config()
    cfg_m.account_equity = float(os.getenv("ACCOUNT_EQUITY", "10000000"))
    sig_fn_m = partial(momentum_signal_fn, bench_df_full=bench_df)
    print("[4/5] momentumバックテスト実行中...")
    res_m = run_walk_forward(ohlcv, trading_dates, {}, sig_fn_m, momentum_exit_fn, cfg_m, min_lookback_days=260)
    trades_to_dataframe(res_m).to_csv(outdir / "backtest_momentum_trades.csv", index=False, encoding="utf-8-sig")
    summary_m = summarize(res_m)
    print("  momentum結果:", summary_m)

    # ---- swing2w ----
    cfg_s = load_swing2w_config()
    cfg_s.account_equity = float(os.getenv("ACCOUNT_EQUITY", "10000000"))
    exit_fn_s = partial(swing2w_exit_fn, time_stop_days=cfg_s.time_stop_days)
    print("[5/5] swing2wバックテスト実行中...")
    res_s = run_walk_forward(ohlcv, trading_dates, {}, swing2w_signal_fn, exit_fn_s, cfg_s, min_lookback_days=260)
    trades_to_dataframe(res_s).to_csv(outdir / "backtest_swing2w_trades.csv", index=False, encoding="utf-8-sig")
    summary_s = summarize(res_s)
    print("  swing2w結果:", summary_s)

    with open(outdir / "backtest_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"バックテスト実行日時: {pd.Timestamp.now()}\n")
        f.write(f"対象期間: {trading_dates[0].date()} 〜 {trading_dates[-1].date()}\n")
        f.write(f"データ被覆率: {meta['data_coverage']*100:.0f}%\n\n")
        f.write("★既知の限界: yfinanceは生存者バイアスを含む(上場廃止銘柄が欠落)。\n")
        f.write("結果は実際より楽観的に出ている可能性がある。\n\n")
        f.write(f"[momentum]\n{summary_m}\n\n")
        f.write(f"[swing2w]\n{summary_s}\n")

    print(f"完了。所要時間: {time.time()-t0:.0f}秒。出力先: {outdir}/")


if __name__ == "__main__":
    main()
