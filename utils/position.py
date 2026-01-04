# utils/position.py
from __future__ import annotations

from typing import Dict, Any

import pandas as pd


def load_positions(path: str) -> pd.DataFrame:
    """
    positions.csv:
      - ticker
      - qty
      - avg_price
    """
    try:
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            return pd.DataFrame(columns=["ticker", "qty", "avg_price"])
        return df
    except Exception:
        return pd.DataFrame(columns=["ticker", "qty", "avg_price"])


def analyze_positions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    ここは最小限（拡張余地あり）
    """
    return {"count": int(len(df))}