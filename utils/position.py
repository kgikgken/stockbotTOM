from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.setup import build_setup_info
from utils.rr_ev import calc_ev
from utils.util import safe_float

def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def analyze_positions(df: pd.DataFrame, mkt_score: int, macro_on: bool) -> Tuple[str, float]:
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry_price = safe_float(row.get("entry_price", 0), 0.0)
        qty = safe_float(row.get("quantity", 0), 0.0)

        cur = entry_price
        hist = None
        try:
            hist = yf.Ticker(ticker).history(period="260d", auto_adjust=True)
            if hist is not None and not hist.empty:
                cur = float(hist["Close"].iloc[-1])
        except Exception:
            pass

        pnl_pct = (cur - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0
        value = cur * qty
        if np.isfinite(value) and value > 0:
            total_value += value

        rr = np.nan
        adj = np.nan
        if hist is not None and hist is not None and len(hist) >= 120:
            info = build_setup_info(hist, macro_on=macro_on)
            ev = calc_ev(info, mkt_score=mkt_score, macro_on=macro_on)
            rr = ev.rr
            adj = ev.adj_ev

        if np.isfinite(rr) and np.isfinite(adj):
            if adj < 0.5:
                lines.append(f"- {ticker}: RR:{rr:.2f} AdjEV:{adj:.2f}（要注意）")
            else:
                lines.append(f"- {ticker}: RR:{rr:.2f} AdjEV:{adj:.2f}")
        else:
            lines.append(f"- {ticker}: 損益 {pnl_pct:+.2f}%")

    if not lines:
        return "ノーポジション", 2_000_000.0

    asset_est = total_value if total_value > 0 else 2_000_000.0
    return "\n".join(lines), float(asset_est)
