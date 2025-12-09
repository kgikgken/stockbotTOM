from __future__ import annotations

import numpy as np
import pandas as pd


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return float("nan")


def calc_vola20(close: pd.Series) -> float:
    """20日ボラ（ATRもどき）"""
    try:
        r = np.log(close / close.shift(1))
        return float(np.nanstd(r.tail(20)))
    except Exception:
        return float("nan")


def compute_rr(hist: pd.DataFrame, mkt_score: int):
    """
    RRエンジンのシンプル版。
    出力:
      rr:      R倍数
      entry:   エントリー価格（直近終値）
      tp_pct:  利確%（+）
      sl_pct:  損切り%（-）
    """
    close = hist["Close"].astype(float)
    entry = _last_val(close)
    if not np.isfinite(entry):
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0)

    vola = calc_vola20(close)
    if (not np.isfinite(vola)) or vola <= 0:
        vola = 0.02

    # stop = 1 ATR, target = 3 ATR
    stop_pct = 1.0 * vola
    target_pct = 3.0 * vola

    rr = target_pct / stop_pct if stop_pct > 0 else 0.0

    return dict(
        rr=float(rr),
        entry=float(entry),
        tp_pct=float(target_pct),
        sl_pct=float(-stop_pct),
    )