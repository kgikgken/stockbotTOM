from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from utils.util import safe_float
from utils.features import add_indicators, atr


def detect_setup(df_raw: pd.DataFrame) -> Dict:
    """
    Setup A/B（順張りのみ）
    A: トレンド押し目（優先）
    B: ブレイク初動（追いかけ禁止 → entry側で制御）
    """
    df = add_indicators(df_raw)
    close = df["Close"].astype(float)
    c = safe_float(close.iloc[-1])

    ma20 = safe_float(df["ma20"].iloc[-1])
    ma50 = safe_float(df["ma50"].iloc[-1])
    slope5 = safe_float(df["ma20_slope5"].iloc[-1])
    rsi = safe_float(df["rsi14"].iloc[-1])
    a = atr(df_raw, 14)
    adv20 = safe_float(df["adv20"].iloc[-1])

    setup = "-"
    setup_reason = ""

    # Setup A: 強トレンド + 押し目（MA20周辺〜MA50寄りも許容）
    if np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma50) and np.isfinite(slope5) and np.isfinite(rsi) and np.isfinite(a):
        trend_ok = (c > ma20 > ma50) and (slope5 > 0)
        pullback_ok = abs(c - ma20) <= 0.8 * a
        rsi_ok = 40 <= rsi <= 62

        if trend_ok and pullback_ok and rsi_ok:
            setup = "A"
            setup_reason = "trend_pullback"

    # Setup B: 20日高値ブレイク（厳選）
    if setup == "-":
        if len(close) >= 25 and np.isfinite(adv20) and np.isfinite(c):
            hh20 = float(close.iloc[-21:-1].max())
            # ブレイク判定（終値）
            if c > hh20:
                setup = "B"
                setup_reason = "breakout_20d"

    return {
        "setup": setup,
        "setup_reason": setup_reason,
        "ma20": ma20,
        "ma50": ma50,
        "rsi14": rsi,
        "atr14": float(a) if np.isfinite(a) else None,
        "adv20": adv20,
    }