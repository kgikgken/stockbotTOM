# utils/diversify.py
from __future__ import annotations

from typing import List, Dict
import numpy as np
import pandas as pd


# -------------------------------------------------
# 相関計算（20日リターン）
# -------------------------------------------------
def calc_corr_matrix(
    price_df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    price_df:
        index = date
        columns = ticker
        values = Close
    """
    if price_df is None or price_df.shape[1] < 2:
        return pd.DataFrame()

    returns = price_df.pct_change().dropna()
    returns = returns.tail(lookback)
    if len(returns) < lookback // 2:
        return pd.DataFrame()

    return returns.corr()


# -------------------------------------------------
# 分散ルール適用
# -------------------------------------------------
def apply_diversification(
    candidates: List[Dict],
    *,
    price_df: pd.DataFrame | None,
    max_per_sector: int = 2,
    corr_threshold: float = 0.75,
    max_final: int = 5,
) -> List[Dict]:
    """
    candidates:
        Sorted by priority already (AdjEV desc, R/day desc, etc)

        必須キー:
          - ticker
          - sector
          - adj_ev
          - r_per_day

    ルール:
      1. 同一セクター最大 max_per_sector
      2. corr > threshold の銘柄は同時採用禁止
      3. 上から順に機械的に採用
    """

    if not candidates:
        return []

    selected: List[Dict] = []
    sector_count: Dict[str, int] = {}

    corr_mat = None
    if price_df is not None and price_df.shape[1] >= 2:
        corr_mat = calc_corr_matrix(price_df)

    for cand in candidates:
        if len(selected) >= max_final:
            break

        ticker = cand.get("ticker")
        sector = cand.get("sector", "UNKNOWN")

        # ---- セクター上限 ----
        cnt = sector_count.get(sector, 0)
        if cnt >= max_per_sector:
            cand["drop_reason"] = "セクター上限"
            continue

        # ---- 相関チェック ----
        high_corr = False
        if corr_mat is not None and ticker in corr_mat.columns:
            for sel in selected:
                t2 = sel.get("ticker")
                if t2 in corr_mat.columns:
                    corr = corr_mat.loc[ticker, t2]
                    if np.isfinite(corr) and corr >= corr_threshold:
                        high_corr = True
                        cand["drop_reason"] = f"相関高({corr:.2f})"
                        break

        if high_corr:
            continue

        # ---- 採用 ----
        selected.append(cand)
        sector_count[sector] = cnt + 1

    return selected


# -------------------------------------------------
# 監視リスト理由付け（落ちた理由を整理）
# -------------------------------------------------
def build_watchlist(
    all_candidates: List[Dict],
    selected: List[Dict],
    *,
    max_watch: int = 10,
) -> List[Dict]:
    """
    採用されなかったが、
    形・RRはOK → 今日は入らない銘柄を整理
    """

    selected_tickers = {c["ticker"] for c in selected}
    watch: List[Dict] = []

    for c in all_candidates:
        if c["ticker"] in selected_tickers:
            continue

        reason = c.get("drop_reason", "")
        if not reason:
            continue

        c2 = c.copy()
        c2["watch_reason"] = reason
        watch.append(c2)

        if len(watch) >= max_watch:
            break

    return watch