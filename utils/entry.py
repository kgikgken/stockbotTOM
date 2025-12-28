from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from utils.util import safe_float, clamp
from utils.features import atr, add_indicators


def calc_entry_zone(df_raw: pd.DataFrame, setup: str) -> Dict:
    """
    IN帯・GU判定・乖離率・行動分類（EXEC_NOW/LIMIT_WAIT/WATCH_ONLY）
    """
    df = add_indicators(df_raw)
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)

    c = safe_float(close.iloc[-1])
    o = safe_float(open_.iloc[-1])
    prev_c = safe_float(close.iloc[-2]) if len(close) >= 2 else c

    a = atr(df_raw, 14)
    if not np.isfinite(a) or a <= 0:
        a = max(c * 0.01, 1.0)

    ma20 = safe_float(df["ma20"].iloc[-1])
    ma50 = safe_float(df["ma50"].iloc[-1])

    if setup == "A":
        center = ma20
        band = 0.5 * a
    elif setup == "B":
        # ブレイクライン = 20日高値（前日まで）
        if len(close) >= 25:
            center = float(close.iloc[-21:-1].max())
        else:
            center = c
        band = 0.3 * a
    else:
        center = c
        band = 0.0

    low = center - band
    high = center + band

    # GU判定（Open > PrevClose + 1.0ATR）
    gu_flag = bool(np.isfinite(o) and np.isfinite(prev_c) and (o > prev_c + 1.0 * a))

    # 乖離率 = distance(Close, center)/ATR
    dist_ratio = float(abs(c - center) / a) if np.isfinite(c) and a > 0 else 999.0

    # 行動分類
    action = "WATCH_ONLY"
    if gu_flag:
        action = "WATCH_ONLY"
    else:
        if (low <= c <= high):
            action = "EXEC_NOW"
        else:
            # 0.8以内なら指値待ち、それ以上は監視
            action = "LIMIT_WAIT" if dist_ratio <= 0.8 else "WATCH_ONLY"

    return {
        "in_center": float(center),
        "in_low": float(low),
        "in_high": float(high),
        "atr": float(a),
        "gu": bool(gu_flag),
        "dist_ratio": float(dist_ratio),
        "action": str(action),
        "price_now": float(c) if np.isfinite(c) else None,
    }