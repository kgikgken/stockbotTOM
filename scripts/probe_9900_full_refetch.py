"""9900.T(サガミホールディングス)の単発フル履歴再取得プローブ（診断専用。
TASKS.mdのタスクではない。store は一切変更しない）。

T-402: store の9900.T系列には2025-01-08を境界とする段差がある（先出し分割適用の
部分適用、inspect_ticker_revision.pyで確認済み）。この段差が「storeが複数回の
fetchをマージした結果生じた artifact」なのか、「yfinanceが今この瞬間に返す
単発のフル履歴取得そのものに含まれる（つまり同じ場所で同じ段差が出る）」のかを
切り分けるため、2600営業日ぶんを1回のyfinance呼び出しで取得し、2025-01-08前後の
Close系列を出力する。cli.step_fetch/step_backfillは経由しない（別プロセス）。
"""
from __future__ import annotations

import warnings

import pandas as pd

warnings.filterwarnings("ignore")

TICKER = "9900.T"
BOUNDARY = pd.Timestamp("2025-01-08")


def main() -> int:
    import yfinance as yf

    df = yf.download(TICKER, period="2600d", progress=False, auto_adjust=False, actions=True)
    if df is None or len(df) == 0:
        print(f"[probe-9900-refetch] {TICKER}: 0行（取得失敗）")
        return 1
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"[probe-9900-refetch] 単発フル履歴取得: 行数={len(df)} "
          f"先頭={df.index[0].date()} 末尾={df.index[-1].date()}")

    window = df[(df.index >= BOUNDARY - pd.Timedelta(days=10))
               & (df.index <= BOUNDARY + pd.Timedelta(days=10))]
    print(f"[probe-9900-refetch] 境界({BOUNDARY.date()})前後のClose:")
    for d, row in window.iterrows():
        print(f"  {d.date()}  Close={float(row['Close']):.2f}  "
              f"Volume={float(row['Volume']):.0f}  "
              f"StockSplits={float(row.get('Stock Splits', 0.0) or 0.0)}")

    ratios = window["Close"].pct_change().add(1.0)
    jump = ratios[(ratios - 1.0).abs() > 0.3]
    if len(jump):
        print(f"[probe-9900-refetch] 境界前後で30%超のギャップを検出: "
              f"{[(d.date(), float(r)) for d, r in jump.items()]}")
    else:
        print("[probe-9900-refetch] 境界前後で30%超のギャップは検出されなかった"
              "（単発取得では段差が無い可能性）")

    splits = df[df.get("Stock Splits", pd.Series(dtype=float)).fillna(0) != 0]
    print(f"[probe-9900-refetch] Stock Splitsイベント記録数={len(splits)}")
    for d, row in splits.iterrows():
        print(f"  {d.date()}: {row['Stock Splits']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
