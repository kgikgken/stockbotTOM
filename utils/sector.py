# utils/sector.py
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "universe_jpx.csv"

# セクター統一カラム候補
SECTOR_COL_CANDIDATES = ["sector", "industry_big", "industry", "sector_name"]


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _load_universe(path: str = UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _infer_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    returns: (ticker_col, sector_col)
    """
    if df is None or df.empty:
        return None, None

    ticker_col = None
    if "ticker" in df.columns:
        ticker_col = "ticker"
    elif "code" in df.columns:
        ticker_col = "code"

    sector_col = None
    for c in SECTOR_COL_CANDIDATES:
        if c in df.columns:
            sector_col = c
            break

    return ticker_col, sector_col


def _fetch_change_pct(ticker: str, days: int = 5) -> float:
    """
    days=5 なら「直近6営業日」から最初と最後で変化率
    """
    try:
        df = yf.Ticker(ticker).history(period=f"{days+1}d", auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return np.nan
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
    except Exception:
        return np.nan


def build_sector_rank_5d(
    top_n: int = 5,
    max_tickers_per_sector: int = 25,
    min_valid: int = 5
) -> Tuple[List[Tuple[str, float]], Dict[str, int], Dict[str, float]]:
    """
    セクターは「選定理由」ではなく、判断補助/制約用。
    - 上位表示: top_n
    - 各セクターの平均5日リターンでランキング
    戻り：
      top_list: [(sector, avg5d), ...]
      rank_map: {sector: rank(1..)}
      score_map:{sector: avg5d}
    """
    uni = _load_universe()
    if uni is None:
        return [], {}, {}

    t_col, s_col = _infer_cols(uni)
    if t_col is None or s_col is None:
        return [], {}, {}

    # セクター別に最大max_tickers_per_sectorだけ取得して平均（計算負荷を抑える）
    sector_scores: Dict[str, float] = {}
    rank_map: Dict[str, int] = {}

    for sec, sub in uni.groupby(s_col):
        sec_name = str(sec).strip() if str(sec).strip() else "不明"
        tickers = sub[t_col].astype(str).tolist()
        if not tickers:
            continue

        tickers = tickers[:max_tickers_per_sector]

        vals = []
        for t in tickers:
            chg = _fetch_change_pct(t, days=5)
            if np.isfinite(chg):
                vals.append(chg)

        if len(vals) < min_valid:
            continue

        sector_scores[sec_name] = float(np.mean(vals))

    if not sector_scores:
        return [], {}, {}

    # rank作成
    sorted_items = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (sec, v) in enumerate(sorted_items, start=1):
        rank_map[sec] = i

    top_list = [(sec, v) for sec, v in sorted_items[:top_n]]
    return top_list, rank_map, sector_scores


def get_sector_rank(sector: str, rank_map: Dict[str, int]) -> int:
    if not sector:
        return 999
    return int(rank_map.get(str(sector), 999))


def sector_constraint_reason(
    sector: str,
    rank_map: Dict[str, int],
    *,
    top_k: int = 5,
    enforce: bool = False
) -> Optional[str]:
    """
    v2方針：
    - セクターは判断補助。選定理由ではない。
    - ただし「制約」として使いたい場合のみ reason を返す。
    enforce=False なら理由は返さない（=落とさない）
    """
    if not enforce:
        return None

    r = get_sector_rank(sector, rank_map)
    if r <= top_k:
        return None
    return f"セクター順位圏外(>{top_k})"