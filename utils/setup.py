from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SetupResult:
    setup: str  # "A1-Strong", "A1", "A2", "B", "NONE"
    trend_strength: float
    pullback_quality: float


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ema_up = up.ewm(alpha=1/period, adjust=False).mean()
    ema_dn = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = ema_up / ema_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def classify_setup(df: pd.DataFrame) -> SetupResult:
    """v2.3 の思想に沿った簡易分類。

    - A1-Strong: Close > SMA20 > SMA50, SMA20上向き, RSI 40-60, SMA20への押しが浅い
    - A1: Close > SMA20 > SMA50, SMA20上向き
    - A2/B: ここでは抑制的に扱う（必要なら後で強化）
    """
    if df is None or len(df) < 60:
        return SetupResult("NONE", 0.0, 0.0)

    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    r = rsi(close, 14)

    last = float(close.iloc[-1])
    s20 = float(sma20.iloc[-1])
    s50 = float(sma50.iloc[-1])

    # TrendStrength: SMA角度 + 位置
    angle = (sma20.iloc[-1] / sma20.iloc[-6] - 1) * 100 if len(sma20.dropna()) >= 6 else 0.0
    pos = 1.0 if (last > s20 > s50) else (0.6 if last > s20 else 0.3)
    trend_strength = float(max(0.2, min(1.4, pos + max(min(angle / 2.0, 0.4), -0.2))))

    # PullbackQuality: SMA20からの乖離が小さいほど良い（押し目）
    dist = abs(last - s20) / max(s20, 1e-9)
    pullback_quality = float(max(0.6, min(1.2, 1.2 - dist * 5)))

    # Setup
    setup = "NONE"
    r_last = float(r.iloc[-1]) if not np.isnan(r.iloc[-1]) else 50.0

    if last > s20 > s50 and angle > -0.2:
        setup = "A1"
        if 40 <= r_last <= 60 and dist <= 0.02 and angle >= 0.0:
            setup = "A1-Strong"

    return SetupResult(setup=setup, trend_strength=trend_strength, pullback_quality=pullback_quality)
