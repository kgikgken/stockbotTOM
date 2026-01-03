# utils/sector.py
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# Config
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
MAX_TICKERS_PER_SECTOR = 25      # 代表性確保（過学習防止）
ADV20_MIN_JPY = 100_000_000      # 表示用の最低流動性（選定理由には使わない）


def _fetch_ohlcv(ticker: str, period: str = "40d") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _pct_5d(df: pd.DataFrame) -> float:
    if df is None or len(df) < 6:
        return float("nan")
    c0 = float(df["Close"].iloc[-6])
    c1 = float(df["Close"].iloc[-1])
    if not np.isfinite(c0) or not np.isfinite(c1) or c0 <= 0:
        return float("nan")
    return float((c1 / c0 - 1.0) * 100.0)


def _adv20_jpy(df: pd.DataFrame) -> float:
    if df is None or "Close" not in df.columns or "Volume" not in df.columns or len(df) < 20:
        return float("nan")
    v = (df["Close"].astype(float) * df["Volume"].astype(float)).rolling(20).mean().iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def _load_universe() -> pd.DataFrame:
    if not os.path.exists(UNIVERSE_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return pd.DataFrame()


def top_sectors_5d(top_n: int = 5) -> List[Tuple[str, float]]:
    """
    5日リターンでのセクター上位表示（判断補助のみ）。
    ・母集団は universe_jpx.csv の全銘柄
    ・各セクターは代表銘柄を最大 MAX_TICKERS_PER_SECTOR 抽出
    ・極端な薄商いは平均算出から除外（表示品質向上）
    戻り：[(sector, avg_5d_pct), ...]
    """
    uni = _load_universe()
    if uni.empty:
        return []

    # column resolve
    if "sector" in uni.columns:
        sec_col = "sector"
    elif "industry_big" in uni.columns:
        sec_col = "industry_big"
    else:
        return []

    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return []

    results: List[Tuple[str, float]] = []

    for sec, sub in uni.groupby(sec_col):
        tickers = [str(x) for x in sub[t_col].dropna().tolist()][:MAX_TICKERS_PER_SECTOR]
        if not tickers:
            continue

        vals = []
        for t in tickers:
            df = _fetch_ohlcv(t, "40d")
            if df is None:
                continue
            adv = _adv20_jpy(df)
            # 表示の質を上げるため、極端な薄商いは除外
            if np.isfinite(adv) and adv < ADV20_MIN_JPY:
                continue
            p = _pct_5d(df)
            if np.isfinite(p):
                vals.append(p)

        if vals:
            results.append((str(sec), float(np.mean(vals))))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def sector_rank_map(top_n: int = 10) -> Dict[str, int]:
    """
    セクター順位マップ（1位が最強）。
    EV補正や説明用に利用。
    """
    ranked = top_sectors_5d(top_n=top_n)
    out: Dict[str, int] = {}
    for i, (sec, _) in enumerate(ranked, start=1):
        out[sec] = i
    return out