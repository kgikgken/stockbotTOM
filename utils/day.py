from __future__ import annotations

import numpy as np
import pandas as pd


def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def score_daytrade_candidate(hist_d: pd.DataFrame, mkt_score: int = 50) -> float:
    """
    デイ用：足の並び＋勢い＋出来高
    0-100
    """
    if hist_d is None or len(hist_d) < 80:
        return 0.0

    df = hist_d.copy()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    c = _last(close)
    m5 = _last(ma5)
    m20 = _last(ma20)
    m60 = _last(ma60)

    ret1 = float((close.iloc[-1] / close.iloc[-2] - 1.0) * 100.0) if len(close) >= 2 else 0.0
    ret5 = float((close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0) if len(close) >= 6 else 0.0

    vma20 = _last((vol * close).rolling(20).mean())
    vnow = _last(vol * close)

    sc = 0.0

    # 足の並び（最重要）
    if np.isfinite(c) and np.isfinite(m5) and np.isfinite(m20) and np.isfinite(m60):
        if c > m5 > m20 > m60:
            sc += 40
        elif c > m5 > m20:
            sc += 30
        elif c > m20 > m60:
            sc += 20

    # 勢い（短期）
    sc += float(np.clip(ret1, -2, 2) * 5)   # 最大±10
    sc += float(np.clip(ret5, -6, 6) * 3)   # 最大±18

    # 出来高（売買代金）
    if np.isfinite(vma20) and vma20 > 0 and np.isfinite(vnow):
        ratio = vnow / vma20
        if ratio >= 1.5:
            sc += 20
        elif ratio >= 1.0:
            sc += 10
        elif ratio <= 0.5:
            sc -= 10

    # 地合いで微調整（デイは地合いの影響大）
    sc += float((mkt_score - 50) * 0.25)

    return float(np.clip(sc, 0, 100))