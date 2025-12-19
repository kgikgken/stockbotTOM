import pandas as pd
import numpy as np
import yfinance as yf
from utils.rr import compute_tp_sl_rr

def load_positions(path: str):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def analyze_positions(df: pd.DataFrame, mkt_score=50):
    if df.empty:
        return "ノーポジション", 2_000_000

    lines = []
    total = 0
    for _, r in df.iterrows():
        t = r["ticker"]
        qty = r["quantity"]
        entry = r["entry_price"]
        cur = yf.Ticker(t).history(period="5d", auto_adjust=True)["Close"].iloc[-1]
        pnl = (cur - entry) / entry * 100
        total += cur * qty
        lines.append(f"- {t}: 損益 {pnl:.2f}%")

    return "\n".join(lines), total