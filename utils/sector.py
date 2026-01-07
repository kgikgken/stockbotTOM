# ============================================
# utils/sector.py
# セクター動向・相対強度計算
# ============================================

from __future__ import annotations

from typing import Dict, List
import pandas as pd


# --------------------------------------------
# セクター5日リターン算出
# --------------------------------------------
def calc_sector_returns(
    df_prices: pd.DataFrame,
    df_universe: pd.DataFrame,
    lookback: int = 5,
) -> Dict[str, float]:
    """
    各セクターの平均リターンを計算
    df_prices: index=date, columns=ticker
    df_universe: ticker, sector を含む
    """
    sector_returns: Dict[str, List[float]] = {}

    for _, row in df_universe.iterrows():
        ticker = row["ticker"]
        sector = row["sector"]

        if ticker not in df_prices.columns:
            continue

        prices = df_prices[ticker].dropna()
        if len(prices) < lookback + 1:
            continue

        ret = prices.iloc[-1] / prices.iloc[-(lookback + 1)] - 1.0
        sector_returns.setdefault(sector, []).append(ret)

    # 平均化
    sector_avg = {
        sec: sum(vals) / len(vals)
        for sec, vals in sector_returns.items()
        if len(vals) > 0
    }

    return sector_avg


# --------------------------------------------
# 上位セクター抽出
# --------------------------------------------
def top_sectors(
    sector_returns: Dict[str, float],
    top_n: int = 5,
) -> List[str]:
    """
    リターン上位のセクター名リストを返す
    """
    return [
        s for s, _ in sorted(
            sector_returns.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
    ]