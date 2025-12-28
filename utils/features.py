from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from utils.util import safe_float


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 3:
        return float("nan")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    v = safe_float(tr.rolling(period).mean().iloc[-1])
    return float(v)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    open_ = out["Open"].astype(float)
    vol = out["Volume"].astype(float) if "Volume" in out.columns else pd.Series(np.nan, index=out.index)

    out["ma20"] = close.rolling(20).mean()
    out["ma50"] = close.rolling(50).mean()
    out["ma10"] = close.rolling(10).mean()
    out["ma5"] = close.rolling(5).mean()

    out["rsi14"] = _rsi(close, 14)

    ret = close.pct_change(fill_method=None)
    out["vola20"] = ret.rolling(20).std()

    out["atr14"] = np.nan
    if len(out) >= 20:
        out["atr14"] = _atr_series(out, 14)

    # 下ヒゲ比率（「完璧すぎない」判定の補助）
    rng = (high - low).replace(0, np.nan)
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    out["lower_shadow_ratio"] = np.where(np.isfinite(rng), lower_shadow / rng, 0.0)

    out["turnover"] = close * vol
    out["adv20"] = out["turnover"].rolling(20).mean()

    # 20MA slope（5日差）
    out["ma20_slope5"] = out["ma20"].pct_change(5, fill_method=None)

    # 60日高値からの距離
    if len(close) >= 60:
        high60 = close.rolling(60).max()
        out["off_high60_pct"] = (close - high60) / (high60 + 1e-9) * 100
    else:
        out["off_high60_pct"] = np.nan

    return out


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rel_strength_20d(stock_close: pd.Series, index_close: pd.Series) -> float:
    """
    RS（相対強度）20日： stock_ret20 - index_ret20
    """
    if stock_close is None or index_close is None:
        return float("nan")
    if len(stock_close) < 21 or len(index_close) < 21:
        return float("nan")

    s = float(stock_close.iloc[-1] / stock_close.iloc[-21] - 1.0)
    i = float(index_close.iloc[-1] / index_close.iloc[-21] - 1.0)
    return float((s - i) * 100.0)


def is_too_perfect(df: pd.DataFrame) -> bool:
    """
    “完璧すぎない” の簡易フィルタ：
      - 直近3日で 1.8ATR 以上の急伸 → 追うな
    """
    if df is None or len(df) < 40:
        return False
    close = df["Close"].astype(float)
    atr14 = atr(df, 14)
    if not np.isfinite(atr14) or atr14 <= 0:
        return False
    if len(close) < 4:
        return False
    surge = float(close.iloc[-1] - close.iloc[-4])
    return surge >= 1.8 * atr14