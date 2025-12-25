import pandas as pd

def load_positions(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def analyze_positions(df):
    if df.empty:
        return "", 2_000_000
    txt = []
    for _, r in df.iterrows():
        txt.append(f"- {r['ticker']}: 損益 {r['pnl']:.2f}%")
    return "\n".join(txt), 2_000_000