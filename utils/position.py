from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def analyze_positions(df: pd.DataFrame, mkt_score: int = 50) -> Tuple[str, float]:
    """
    positions.csv expects at least ticker, entry_price, quantity (optional).
    If missing or empty -> returns "ノーポジション", 2,000,000
    """
    if df is None or df.empty:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry_price = float(row.get("entry_price", 0) or 0)
        qty = float(row.get("quantity", 0) or 0)

        # current
        cur = entry_price if entry_price > 0 else 0.0
        try:
            h = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if h is not None and not h.empty:
                cur = float(h["Close"].iloc[-1])
        except Exception:
            pass

        pnl = "n/a"
        if entry_price > 0 and np.isfinite(cur):
            pnl_pct = (cur - entry_price) / entry_price * 100.0
            pnl = f"{pnl_pct:.2f}%"

        value = cur * qty if (np.isfinite(cur) and qty > 0) else 0.0
        if np.isfinite(value) and value > 0:
            total_value += value

        if pnl != "n/a":
            lines.append(f"- {ticker}: 損益 {pnl}")
        else:
            lines.append(f"- {ticker}: 損益 n/a")

    if not lines:
        return "ノーポジション", 2_000_000.0

    asset_est = total_value if total_value > 0 else 2_000_000.0
    return "\n".join(lines), float(asset_est)