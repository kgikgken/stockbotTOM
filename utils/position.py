from __future__ import annotations
import pandas as pd
import numpy as np


def load_positions(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def analyze_positions(df: pd.DataFrame, mkt_score: int):
    """
    positions.csv を解析して
    - LINE用ポジション文字列
    - 総資産推定
    を返す
    """
    if df is None or len(df) == 0:
        text = "ノーポジション"
        asset = 2_000_000.0
        return text, asset

    lines = []
    total = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        entry = float(row.get("entry_price", 0.0))
        qty = float(row.get("quantity", 0.0))
        price = float(row.get("current_price", entry))

        pnl_pct = (price - entry) / entry * 100.0 if entry > 0 else 0.0
        value = qty * price
        total += value

        lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")

    if total <= 0 or not np.isfinite(total):
        total = 2_000_000.0

    text = "\n".join(lines) if lines else "ノーポジション"
    return text, float(total)