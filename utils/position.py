import pandas as pd

def load_positions(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def analyze_positions(df, mkt_score):
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000

    lines = []
    total = 0
    for _, r in df.iterrows():
        lines.append(f"- {r.get('ticker')}: 損益 {r.get('pnl_pct', 0):.2f}%")
        total += r.get("value", 0)

    return "\n".join(lines), total if total > 0 else 2_000_000