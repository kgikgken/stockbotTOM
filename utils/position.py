from __future__ import annotations

import pandas as pd
import yfinance as yf


POSITIONS_PATH = "positions.csv"


def load_positions():
    try:
        return pd.read_csv(POSITIONS_PATH)
    except Exception:
        return pd.DataFrame()


def analyze_positions() -> str:
    df = load_positions()
    if df.empty:
        return "ノーポジション"

    lines = []
    for _, r in df.iterrows():
        ticker = r["ticker"]
        entry = r["entry_price"]
        qty = r["quantity"]

        try:
            cur = yf.download(ticker, period="5d", auto_adjust=True)["Close"].iloc[-1]
            pnl = (cur - entry) / entry * 100
            lines.append(f"- {ticker}: 損益 {pnl:.2f}%")
        except Exception:
            lines.append(f"- {ticker}: 取得失敗")

    return "\n".join(lines)