from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .rr import compute_tp_sl_rr

def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def analyze_positions(df: pd.DataFrame, mkt_score: int = 50) -> Tuple[str, float]:
    """Return (pos_text, total_asset_est).
    positions.csv expects ticker, entry_price, quantity but must not fail if missing.
    """
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry_price = float(row.get("entry_price", 0) or 0)
        qty = float(row.get("quantity", 0) or 0)

        cur = entry_price
        try:
            h = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if h is not None and not h.empty:
                cur = float(h["Close"].iloc[-1])
        except Exception:
            pass

        pnl_pct = (cur - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0
        value = cur * qty
        if np.isfinite(value) and value > 0:
            total_value += value

        rr = 0.0
        try:
            hist = yf.Ticker(ticker).history(period="260d", auto_adjust=True)
            if hist is not None and len(hist) >= 120:
                rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score, for_day=False)
                rr = float(rr_info.get("rr", 0.0))
        except Exception:
            rr = 0.0

        if rr > 0:
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}% RR:{rr:.2f}R")
        else:
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")

    if not lines:
        return "ノーポジション", 2_000_000.0

    asset_est = total_value if total_value > 0 else 2_000_000.0
    return "\n".join(lines), float(asset_est)
