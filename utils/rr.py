from __future__ import annotations

import numpy as np
import pandas as pd


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def calc_vola20(close: pd.Series) -> float:
    """20日分の対数リターンの標準偏差"""
    try:
        r = np.log(close / close.shift(1))
        return float(np.nanstd(r.tail(20)))
    except Exception:
        return np.nan


def compute_rr(hist: pd.DataFrame, mkt_score: int = 50) -> dict:
    """
    シンプルな ATR ベースRR（将来の拡張用）
    """
    close = hist["Close"].astype(float)
    entry = _last_val(close)
    if not np.isfinite(entry):
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0)

    vola = calc_vola20(close)
    if not np.isfinite(vola) or vola <= 0:
        vola = 0.02

    stop_pct = 1.0 * vola
    target_pct = 3.0 * vola
    rr = target_pct / stop_pct if stop_pct > 0 else 0.0

    return dict(
        rr=float(rr),
        entry=float(entry),
        tp_pct=float(target_pct * 100),
        sl_pct=float(-stop_pct * 100),
    )