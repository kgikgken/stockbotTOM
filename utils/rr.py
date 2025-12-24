from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

SetupType = Literal["A", "B"]


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
        axis=1,
    ).max(axis=1)

    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def _sma(series: pd.Series, window: int) -> float:
    if series is None or len(series) < window:
        return _last(series)
    v = series.rolling(window).mean().iloc[-1]
    return float(v) if np.isfinite(v) else _last(series)


def _hh(df: pd.DataFrame, window: int) -> float:
    if df is None or len(df) < window:
        return float("nan")
    v = df["High"].astype(float).rolling(window).max().iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def _ll(df: pd.DataFrame, window: int) -> float:
    if df is None or len(df) < window:
        return float("nan")
    v = df["Low"].astype(float).rolling(window).min().iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


@dataclass(frozen=True)
class TradePlan:
    setup: SetupType
    atr: float
    in_center: float
    in_low: float
    in_high: float
    stop: float
    tp1: float
    tp2: float
    r: float
    expected_days: float
    r_per_day: float
    gu_flag: bool
    in_distance_atr: float
    action: str  # 即IN可 / 指値待ち / 監視のみ


def classify_setup(hist: pd.DataFrame) -> Optional[SetupType]:
    """
    SetupType A: トレンド押し目
      - Close > SMA20 > SMA50
      - SMA20傾き>0（5日前比較）
      - Close が SMA20 に接近（0.8*ATR以内）
    SetupType B: ブレイク
      - Close > HH20
      - 出来高: Vol >= 1.5*VolSMA20
    """
    if hist is None or len(hist) < 120:
        return None

    df = hist.copy()
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    close = _last(c)
    atr = _atr(df, 14)
    if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
        return None

    sma20 = _sma(c, 20)
    sma50 = _sma(c, 50)
    sma20_prev5 = float(c.rolling(20).mean().iloc[-6]) if len(c) >= 26 else float("nan")
    slope20 = (sma20 - sma20_prev5) if np.isfinite(sma20) and np.isfinite(sma20_prev5) else float("nan")

    # A
    if np.isfinite(sma20) and np.isfinite(sma50) and close > sma20 > sma50:
        if np.isfinite(slope20) and slope20 > 0:
            if abs(close - sma20) <= 0.8 * atr:
                return "A"

    # B
    hh20 = _hh(df, 20)
    if np.isfinite(hh20) and close > hh20 * 0.999:
        vol = _last(v)
        vol_sma20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float("nan")
        if np.isfinite(vol) and np.isfinite(vol_sma20) and vol_sma20 > 0 and vol >= 1.5 * vol_sma20:
            return "B"

    return None


def build_trade_plan(
    hist: pd.DataFrame,
    setup: SetupType,
    today_open: Optional[float] = None,
    prev_close: Optional[float] = None,
) -> Optional[TradePlan]:
    if hist is None or len(hist) < 120:
        return None

    df = hist.copy()
    close_s = df["Close"].astype(float)

    close = _last(close_s)
    atr = _atr(df, 14)
    if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
        return None

    sma20 = _sma(close_s, 20)
    hh20 = _hh(df, 20)
    ll10 = _ll(df, 10)
    ll12 = _ll(df, 12)

    if today_open is None:
        try:
            today_open = float(df["Open"].astype(float).iloc[-1])
        except Exception:
            today_open = close
    if prev_close is None:
        try:
            prev_close = float(close_s.iloc[-2])
        except Exception:
            prev_close = close

    gu_flag = bool(np.isfinite(today_open) and np.isfinite(prev_close) and (today_open > prev_close + 1.0 * atr))

    if setup == "A":
        in_center = float(sma20)
        in_low = in_center - 0.5 * atr
        in_high = in_center + 0.5 * atr
        stop = in_low - 0.7 * atr
        if np.isfinite(ll12):
            stop = min(stop, ll12 - 0.2 * atr)
    else:
        if not np.isfinite(hh20):
            return None
        in_center = float(hh20)
        in_low = in_center - 0.3 * atr
        in_high = in_center + 0.3 * atr
        stop = in_center - 1.0 * atr
        if np.isfinite(ll10):
            stop = min(stop, ll10 - 0.2 * atr)

    risk = max(0.01, in_center - stop)
    tp2 = in_center + 3.0 * risk
    tp1 = in_center + 1.5 * risk
    r = (tp2 - in_center) / risk if risk > 0 else 0.0

    expected_days = (tp2 - in_center) / atr if atr > 0 else 99.0
    expected_days = float(np.clip(expected_days, 0.5, 20.0))
    r_per_day = float(r / expected_days) if expected_days > 0 else 0.0

    in_dist_atr = abs(close - in_center) / atr if atr > 0 else 99.0

    if gu_flag:
        action = "監視のみ"
    elif in_dist_atr > 0.8:
        action = "監視のみ"
    elif close > in_high:
        action = "指値待ち"
    else:
        action = "即IN可"

    return TradePlan(
        setup=setup,
        atr=float(atr),
        in_center=float(in_center),
        in_low=float(in_low),
        in_high=float(in_high),
        stop=float(stop),
        tp1=float(tp1),
        tp2=float(tp2),
        r=float(r),
        expected_days=float(expected_days),
        r_per_day=float(r_per_day),
        gu_flag=bool(gu_flag),
        in_distance_atr=float(in_dist_atr),
        action=str(action),
    )
