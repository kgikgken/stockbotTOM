from __future__ import annotations

import numpy as np
import pandas as pd


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def calc_vola20(close: pd.Series) -> float:
    try:
        r = np.log(close / close.shift(1))
        v = float(np.nanstd(r.tail(20)))
        if not np.isfinite(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def rr_min_by_market(mkt_score: int) -> float:
    if mkt_score >= 70:
        return 1.8
    if mkt_score >= 60:
        return 2.0
    if mkt_score >= 50:
        return 2.2
    if mkt_score >= 40:
        return 2.5
    return 3.0


def compute_rr(hist: pd.DataFrame, mkt_score: int, in_rank: str | None = None) -> dict:
    """
    出力: dict
      rr:     R倍数
      entry:  エントリー価格
      tp_pct: 利確% (0.10 = +10%)
      sl_pct: 損切% (-0.04 = -4%)
    """
    close = hist["Close"].astype(float)
    entry = _last_val(close)
    if not np.isfinite(entry) or entry <= 0:
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0)

    vola = calc_vola20(close)
    if not np.isfinite(vola) or vola <= 0:
        vola = 0.02

    # ベース: ATR型
    stop_pct = vola * 1.5
    tp_pct = vola * 4.0

    # 地合い補正
    if mkt_score >= 70:
        tp_pct *= 1.2
    elif mkt_score < 45:
        tp_pct *= 0.9
        stop_pct *= 0.9

    # INランク補正
    if in_rank == "強IN":
        tp_pct *= 1.1
        stop_pct *= 0.9
    elif in_rank == "弱めIN":
        tp_pct *= 0.9
        stop_pct *= 0.9

    # クリップ
    tp_pct = float(np.clip(tp_pct, 0.06, 0.30))
    stop_pct = float(np.clip(stop_pct, 0.02, 0.12))

    rr = tp_pct / stop_pct if stop_pct > 0 else 0.0

    return dict(
        rr=float(rr),
        entry=float(entry),
        tp_pct=float(tp_pct),
        sl_pct=float(-stop_pct),
    )