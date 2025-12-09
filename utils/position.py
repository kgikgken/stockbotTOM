import pandas as pd


def load_positions(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def analyze_positions(df: pd.DataFrame):
    """
    positions.csv を解析して
    - LINE用のポジション文字列
    - 総資産推定
    を返す

    必ず (text, asset) の 2値で return する
    """
    if df is None or len(df) == 0:
        text = "ノーポジション"
        asset = 2_000_000
        return text, asset

    lines = []
    total = 0.0

    for _, row in df.iterrows():
        ticker = row.get("ticker", "")
        entry = float(row.get("entry_price", 0))
        qty = float(row.get("quantity", 0))
        price = float(row.get("current_price", entry))
        pnl_pct = (price - entry) / entry * 100 if entry > 0 else 0

        value = qty * price
        total += value

        lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")

    text = "\n".join(lines) if lines else "ノーポジション"

    if total <= 0:
        total = 2_000_000

    return text, float(total)