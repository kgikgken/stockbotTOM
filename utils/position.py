import pandas as pd
import yfinance as yf

def load_positions(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

def analyze_positions(df, mkt_score=50):
    if df.empty:
        return "ノーポジション", 2_000_000

    lines = []
    total = 0
    for _, r in df.iterrows():
        t = r["ticker"]
        e = r["entry_price"]
        q = r["quantity"]
        cur = yf.Ticker(t).history(period="5d", auto_adjust=True)["Close"].iloc[-1]
        pnl = (cur - e) / e * 100
        total += cur * q
        lines.append(f"- {t}: 損益 {pnl:.2f}%")

    return "\n".join(lines), total