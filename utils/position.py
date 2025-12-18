from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from utils.rr import compute_tp_sl_rr

def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def analyze_positions(df: pd.DataFrame, mkt_score: int):
    if df.empty:
        return "ノーポジション", 2_000_000.0

    lines = []
    total = 0.0

    for _, r in df.iterrows():
        t = r["ticker"]
        entry = r["entry_price"]
        qty = r["quantity"]

        try:
            h = yf.Ticker(t).history(period="5d", auto_adjust=True)
            cur = h["Close"].iloc[-1]
        except Exception:
            cur = entry

        pnl = (cur-entry)/entry*100 if entry>0 else 0
        total += cur*qty

        rr = compute_tp_sl_rr(
            yf.Ticker(t).history(period="200d", auto_adjust=True),
            mkt_score
        )["rr"]

        lines.append(f"- {t}: 損益 {pnl:.2f}% RR:{rr:.2f}R")

    return "\n".join(lines), float(total or 2_000_000)