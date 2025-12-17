
from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import add_indicators


def trend_gate(hist: pd.DataFrame) -> tuple[bool, dict]:
    """
    逆張り排除ゲート（必須）
    - Close > MA20 > MA50
    - MA20 上向き（5日前比+）
    - 安値切り上げ（low[-1] > low[-3] > low[-5]）
    - 20日平均売買代金 >= 1億
    """
    df = add_indicators(hist)
    diag = {"reason": ""}

    if df is None or len(df) < 60:
        diag["reason"] = "data_short"
        return False, diag

    c = df["Close"].astype(float)
    ma20 = df["ma20"].astype(float)
    ma50 = df["ma50"].astype(float)
    low = df["Low"].astype(float) if "Low" in df.columns else None
    t20 = df["turnover_avg20"].astype(float)

    c_last = float(c.iloc[-1])
    ma20_last = float(ma20.iloc[-1])
    ma50_last = float(ma50.iloc[-1])
    ma20_prev5 = float(ma20.iloc[-6]) if len(ma20) >= 6 else float("nan")

    if not (np.isfinite(c_last) and np.isfinite(ma20_last) and np.isfinite(ma50_last)):
        diag["reason"] = "nan"
        return False, diag

    if not (c_last > ma20_last > ma50_last):
        diag["reason"] = "not_above_mas"
        return False, diag

    if not (np.isfinite(ma20_prev5) and ma20_last > ma20_prev5):
        diag["reason"] = "ma20_not_up"
        return False, diag

    if low is None or len(low) < 6:
        diag["reason"] = "low_missing"
        return False, diag

    if not (float(low.iloc[-1]) > float(low.iloc[-3]) > float(low.iloc[-5])):
        diag["reason"] = "lows_not_rising"
        return False, diag

    t_last = float(t20.iloc[-1])
    if not (np.isfinite(t_last) and t_last >= 1e8):
        diag["reason"] = "illiquid"
        return False, diag

    diag["reason"] = "ok"
    return True, diag
