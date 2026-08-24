"""指数ティッカーの疎通確認用ワンショット・スクリプト（TASKS.mdのタスクではない）。

^TPX（TOPIX）が実際に yfinance で取得できるか、代替候補（^TOPX, 1306.T等）で
取れるものがあるかを GitHub Actions 上（実インターネット環境）で確認する。
本番の取得ロジック（yf_fetch.fetch_index）には影響しない、診断専用。
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

CANDIDATES = ["^TPX", "^TOPX", "1306.T", "1348.T", "1475.T", "998405.T", "^N225"]


def main() -> None:
    import yfinance as yf

    for ticker in CANDIDATES:
        try:
            df = yf.download(ticker, period="15d", progress=False, auto_adjust=False)
        except Exception as e:  # noqa: BLE001
            print(f"[probe] {ticker}: EXCEPTION {e!r}")
            continue
        if df is None or len(df) == 0:
            print(f"[probe] {ticker}: 0行（取得失敗）")
            continue
        last = df.index[-1]
        close_col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        last_close = float(close_col.iloc[-1]) if hasattr(close_col, "iloc") else float(close_col.tail(1))
        print(f"[probe] {ticker}: 行数={len(df)} 最終日={last} 終値={last_close}")


if __name__ == "__main__":
    main()
