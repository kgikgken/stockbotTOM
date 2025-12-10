from __future__ import annotations
import pandas as pd
import numpy as np


def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def analyze_positions(df: pd.DataFrame):
    """
    returns:
      pos_text: str
      total_asset: float
    """
    if df is None or len(df) == 0:
        return "ノーポジション", 3_000_000.0

    lines = []
    total = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        entry = float(row.get("entry_price", 0) or 0)
        qty = float(row.get("quantity", 0) or 0)
        price = float(row.get("current_price", entry) or entry)

        if entry > 0:
            pnl_pct = (price - entry) / entry * 100.0
        else:
            pnl_pct = 0.0

        value = qty * price
        total += value

        lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")

    if total <= 0 or not np.isfinite(total):
        total = 3_000_000.0

    text = "\n".join(lines) if lines else "ノーポジション"
    return text, float(total)