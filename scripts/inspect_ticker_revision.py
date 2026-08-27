"""特定銘柄のstore内のrevisions(改訂履歴)とClose系列の該当期間を調べる
（診断専用。9900.Tのd2 no-op不一致の原因調査、TASKS.mdのタスクではない）。

store を変更しない（読み取りのみ）。
"""
from __future__ import annotations

import sys

import pandas as pd


def main() -> int:
    import argparse

    from stockbot.config import Settings
    from stockbot.data.store import OhlcvStore, from_long

    ap = argparse.ArgumentParser(prog="python scripts/inspect_ticker_revision.py")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--around-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--window-days", type=int, default=15)
    args = ap.parse_args()

    cfg = Settings.from_env()
    store = OhlcvStore(cfg.store_dir, cfg.daily_dir)
    ohlcv = from_long(store.load())
    df = ohlcv.get(args.ticker)
    if df is None:
        print(f"[NG] {args.ticker} が store に無い")
        return 1

    center = pd.Timestamp(args.around_date)
    lo = center - pd.Timedelta(days=args.window_days)
    hi = center + pd.Timedelta(days=args.window_days)
    window = df[(df.index >= lo) & (df.index <= hi)]
    print(f"=== {args.ticker} の Close/Stock Splits ({lo.date()} 〜 {hi.date()}) ===")
    print(window[["Close", "Stock Splits"]].to_string())

    if store.revisions_path.exists():
        rev = pd.read_csv(store.revisions_path)
        if len(rev) == 0:
            print("\nrevisions.csv.gz は空（改訂履歴なし）")
        else:
            rev["date"] = pd.to_datetime(rev["date"], format="mixed")
            rev_t = rev[rev["ticker"] == args.ticker]
            rev_t_window = rev_t[(rev_t["date"] >= lo - pd.Timedelta(days=400))
                                 & (rev_t["date"] <= hi)]
            print(f"\n=== {args.ticker} の改訂履歴（revisions.csv.gz、前後400日+窓） ===")
            if len(rev_t_window):
                print(rev_t_window.to_string())
            else:
                print("該当する改訂履歴なし")
    else:
        print("\nrevisions.csv.gz が無い")

    splits_in_full_history = df[df["Stock Splits"].fillna(0) != 0]
    print(f"\n=== {args.ticker} の全履歴中の分割イベント ===")
    if len(splits_in_full_history):
        print(splits_in_full_history[["Close", "Stock Splits"]].to_string())
    else:
        print("分割イベントなし（Stock Splits列に非ゼロが無い）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
