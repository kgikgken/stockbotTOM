# ============================================
# utils/sector.py
# セクター相対強度算出
# ============================================

import pandas as pd


# --------------------------------------------
# セクター5日リターン算出
# --------------------------------------------
def calc_sector_returns(
    stock_data: dict,
    meta_data: dict,
    lookback: int = 5,
):
    """
    各セクターの短期リターンを算出
    stock_data: { code: df }
    meta_data: { code: meta }
    """

    sector_returns = {}

    for code, df in stock_data.items():
        meta = meta_data.get(code)
        if meta is None:
            continue

        sector = meta.get("sector")
        if not sector:
            continue

        if len(df) < lookback + 1:
            continue

        try:
            ret = (
                df["Close"].iloc[-1] /
                df["Close"].iloc[-(lookback + 1)] - 1.0
            )
        except Exception:
            continue

        sector_returns.setdefault(sector, []).append(ret)

    # セクター平均
    sector_avg = {}
    for sector, rets in sector_returns.items():
        if len(rets) == 0:
            continue
        sector_avg[sector] = sum(rets) / len(rets)

    return sector_avg


# --------------------------------------------
# セクター順位付け
# --------------------------------------------
def rank_sectors(sector_returns: dict):
    """
    セクターをリターン順に順位付け
    return:
        sector_ranks: { sector: rank }
        sorted_list: [(sector, return)]
    """

    sorted_sectors = sorted(
        sector_returns.items(),
        key=lambda x: x[1],
        reverse=True
    )

    sector_ranks = {}
    for i, (sector, _) in enumerate(sorted_sectors, start=1):
        sector_ranks[sector] = i

    return sector_ranks, sorted_sectors


# --------------------------------------------
# 上位セクター抽出
# --------------------------------------------
def top_sectors(
    sector_returns: dict,
    top_n: int = 5
):
    """
    上位セクターのみ返す
    """

    sorted_sectors = sorted(
        sector_returns.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_sectors[:top_n]