# ============================================
# utils/sector.py
# セクター5日リターン集計（判断補助）
# ============================================

from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd


def top_sectors_5d(universe_df: pd.DataFrame, returns_5d: Dict[str, float], top_n: int = 5) -> List[Tuple[str, float]]:
    """
    returns_5d: ticker->5d return(%)
    universe_df: ticker, sector を含む想定
    """
    if universe_df is None or universe_df.empty:
        return []

    df = universe_df.copy()
    if "ticker" not in df.columns:
        return []
    if "sector" not in df.columns:
        df["sector"] = "不明"

    df["ret5d"] = df["ticker"].map(returns_5d).fillna(0.0)
    g = df.groupby("sector")["ret5d"].mean().sort_values(ascending=False)
    out = [(str(k), float(v)) for k, v in g.head(top_n).items()]
    return out