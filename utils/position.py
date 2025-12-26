# utils/position.py
import pandas as pd
import yfinance as yf

def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def analyze_positions(df: pd.DataFrame, mkt_score: int):
    if df.empty:
        return "ノーポジション", 2_000_000
    lines = []
    total = 0
    for _, r in df.iterrows():
        t = r["ticker"]
        entry = r["entry_price"]
        qty = r["quantity"]
        h = yf.Ticker(t).history(period="5d", auto_adjust=True)
        cur = h["Close"].iloc[-1] if len(h) else entry
        pnl = (cur - entry) / entry * 100
        total += cur * qty
        lines.append(f"- {t}: 損益 {pnl:.2f}%")
    return "\n".join(lines), total