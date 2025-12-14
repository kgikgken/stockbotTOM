from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import pandas as pd


RunnerKind = Literal["A1_breakout", "A2_prebreak", "none"]


def _last(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return float("nan")


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return float("nan")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def _rolling_high(close: pd.Series, window: int) -> float:
    if close is None or len(close) < 5:
        return float("nan")
    w = min(window, len(close))
    return float(close.tail(w).max())


def qualify_runner_grade(hist: pd.DataFrame) -> Dict:
    """
    vAB' Prime: Runner判定（A1/A2）+ strength(0-100) + grade(1-3)

    A1_breakout:
      - MA20>MA60（上昇基調）
      - 60日高値の1.5%以内（ブレイク近傍）
      - 20日売買代金 >= 1億/日

    A2_prebreak:
      - MA20>MA60
      - 高値からの押し戻し 0.3〜1.2ATR
      - RSI<=75（過熱除外）
      - 直近安値切り上げ
    """
    if hist is None or len(hist) < 90:
        return dict(is_runner=False, kind="none", strength=0.0, grade=1)

    df = hist.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    c = _last(close)
    if not np.isfinite(c) or c <= 0:
        return dict(is_runner=False, kind="none", strength=0.0, grade=1)

    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else c
    ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else ma20
    trend_ok = bool(np.isfinite(ma20) and np.isfinite(ma60) and ma20 > ma60)

    atr = _atr(df, 14)
    if not np.isfinite(atr) or atr <= 0:
        atr = c * 0.015

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi14 = 100 - (100 / (1 + rs))
    rsi = _last(rsi14)

    turnover = close * vol
    t20 = float(turnover.rolling(20).mean().iloc[-1]) if len(turnover) >= 20 else float("nan")
    liq_ok = bool(np.isfinite(t20) and t20 >= 1e8)

    high60 = _rolling_high(close, 60)
    dist_to_high = (high60 / c - 1.0) * 100.0 if np.isfinite(high60) else 999.0

    recent_high = float(high.iloc[-6:-1].max()) if len(high) >= 6 else float(high.tail(5).max())
    pullback = float(recent_high - c)
    pullback_atr = pullback / atr if atr > 0 else 999.0

    reup = bool(len(low) >= 3 and float(low.iloc[-1]) > float(low.iloc[-2]))

    a1 = trend_ok and liq_ok and (dist_to_high <= 1.5) and (c >= ma20 * 0.995)
    a2 = trend_ok and liq_ok and (0.3 <= pullback_atr <= 1.2) and reup and (np.isfinite(rsi) and rsi <= 75)

    kind: RunnerKind = "none"
    is_runner = False
    strength = 0.0

    if a1:
        kind = "A1_breakout"
        is_runner = True
        strength = 55.0
        strength += float(np.clip(20.0 - dist_to_high * 10.0, 0.0, 20.0))
        strength += 15.0 if c > ma20 else 5.0
        strength += 10.0 if t20 >= 5e8 else float(np.clip((t20 - 1e8) / 4e8 * 10.0, 0.0, 10.0))
    elif a2:
        kind = "A2_prebreak"
        is_runner = True
        strength = 50.0
        strength += float(np.clip((1.2 - pullback_atr) * 15.0, 0.0, 15.0))
        strength += 10.0 if reup else 0.0
        strength += 10.0 if t20 >= 5e8 else float(np.clip((t20 - 1e8) / 4e8 * 10.0, 0.0, 10.0))
        if np.isfinite(rsi):
            strength += float(np.clip((65.0 - rsi) * 0.3, 0.0, 8.0))

    strength = float(np.clip(strength, 0.0, 100.0))

    if strength >= 75:
        grade = 3
    elif strength >= 60:
        grade = 2
    else:
        grade = 1

    return dict(is_runner=is_runner, kind=kind, strength=strength, grade=int(grade))
