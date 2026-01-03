from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import safe_float


def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def analyze_positions(df: pd.DataFrame, mkt_score: int = 50) -> Tuple[str, float]:
    """
    positions.csv は最低限 ticker, entry_price, quantity を想定（無くても落ちない）
    total_asset は「保有評価額」推定（ノーポジなら 2,000,000）
    """
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry_price = safe_float(row.get("entry_price", 0) or 0, 0.0)
        qty = safe_float(row.get("quantity", 0) or 0, 0.0)

        cur = entry_price
        try:
            h = yf.Ticker(ticker).history(period="10d", auto_adjust=True)
            if h is not None and not h.empty:
                cur = safe_float(h["Close"].iloc[-1], entry_price)
        except Exception:
            pass

        pnl_pct = (cur - entry_price) / entry_price * 100.0 if entry_price > 0 else float("nan")
        value = cur * qty
        if np.isfinite(value) and value > 0:
            total_value += value

        if np.isfinite(pnl_pct):
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")
        else:
            lines.append(f"- {ticker}: 損益 n/a")

    if not lines:
        return "ノーポジション", 2_000_000.0

    asset_est = total_value if total_value > 0 else 2_000_000.0
    return "\n".join(lines), float(asset_est)