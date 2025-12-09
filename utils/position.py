from __future__ import annotations

from typing import Tuple

import pandas as pd
import numpy as np


def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def analyze_positions(df: pd.DataFrame) -> Tuple[str, float]:
    """
    positions.csv を解析して
    - LINE用のポジション文字列
    - 総資産推定
    を返す
    """
    if df is None or len(df) == 0:
        text = "ノーポジション"
        asset = 3_000_000.0  # デフォルト運用規模（調整可）
        return text, float(asset)

    lines = []
    total = 0.0

    for _, row in df.iterrows():
        ticker = row.get("ticker", "")
        entry = float(row.get("entry_price", 0) or 0)
        qty = float(row.get("quantity", 0) or 0)
        price = float(row.get("current_price", entry) or entry)
        pnl_pct = (price - entry) / entry * 100 if entry > 0 else 0.0

        value = qty * price
        total += value

        lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")

    text = "\n".join(lines) if lines else "ノーポジション"

    if total <= 0:
        total = 3_000_000.0

    return text, float(total)