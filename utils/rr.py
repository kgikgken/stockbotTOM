from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import calc_inout_for_stock


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def compute_rr(hist: pd.DataFrame, mkt_score: int) -> dict:
    """
    戻り値:
      rr:      R倍数
      entry:   エントリー価格（現値ベース）
      tp_pct:  利確%（0.15 → +15%）
      sl_pct:  損切り%（0.05 → -5%）
      in_rank: "強IN" / "通常IN" / "弱めIN" / "様子見"
    """
    close = hist["Close"].astype(float)
    entry = _last_val(close)
    if not np.isfinite(entry) or entry <= 0:
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0, in_rank="様子見")

    # INランク & TP/SL（％）
    in_rank, tp_pct, sl_pct = calc_inout_for_stock(hist)  # tp/sl は %（絶対値）

    # 地合いでTP/SL微調整
    # 強い地合い → TP少し伸ばす
    if mkt_score >= 70:
        tp_pct *= 1.2
    elif mkt_score <= 40:
        tp_pct *= 0.9
        sl_pct *= 0.9

    # % → 小数
    tp_frac = float(tp_pct) / 100.0
    sl_frac = float(sl_pct) / 100.0

    rr = tp_frac / sl_frac if sl_frac > 0 else 0.0

    # 上限・下限クリップ
    rr = float(np.clip(rr, 0.0, 6.0))

    return dict(
        rr=rr,
        entry=float(entry),
        tp_pct=tp_frac,
        sl_pct=sl_frac,
        in_rank=in_rank,
    )