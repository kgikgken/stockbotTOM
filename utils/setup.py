from __future__ import annotations

import numpy as np
import pandas as pd

from utils.features import sma, rsi, atr


def detect_setup_type(df: pd.DataFrame) -> str:
    """
    A: トレンド押し目
    B: ブレイク初動（厳選）
    """
    if df is None or len(df) < 80:
        return "-"

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    c = float(close.iloc[-1])
    ma20 = sma(close, 20)
    ma50 = sma(close, 50)
    r = rsi(close, 14)
    a = atr(df, 14)
    if not np.isfinite(a) or a <= 0:
        return "-"

    # A: MA構造 + 押し目(20-50付近) + 過熱なし
    if np.isfinite(ma20) and np.isfinite(ma50) and c > ma20 > ma50 and 35 <= r <= 70:
        # 押し目：|Close - MA20| <= 0.8ATR
        if abs(c - ma20) <= 0.8 * a:
            return "A"

    # B: 20日高値ブレイク + 出来高増（追いかけ禁止なので“形だけ”）
    if len(close) >= 25:
        hh20 = float(high.iloc[-21:-1].max())
        vol_ma20 = float((vol * close).rolling(20).mean().iloc[-1]) if len(close) >= 20 else np.nan
        value_now = float(vol.iloc[-1] * close.iloc[-1]) if np.isfinite(vol.iloc[-1]) else np.nan
        if c > hh20 and np.isfinite(value_now) and np.isfinite(vol_ma20) and value_now >= 1.5 * vol_ma20:
            return "B"

    return "-"