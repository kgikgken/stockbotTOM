# ============================================
# utils/features.py
# テクニカル特徴量（トレンド/押し目/ボラ/流動性）
# ============================================

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd


# -----------------------------
# helpers
# -----------------------------
def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ma_up = up.rolling(n, min_periods=n).mean()
    ma_down = down.rolling(n, min_periods=n).mean()
    rs = ma_up / ma_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(n, min_periods=n).mean()


def _last(x: pd.Series, default: float = np.nan) -> float:
    try:
        v = x.iloc[-1]
        return float(v) if pd.notna(v) else float(default)
    except Exception:
        return float(default)


def _safe_pct(a: float, b: float) -> float:
    if b is None or b == 0 or np.isnan(b):
        return 0.0
    return float(a / b)


# --------------------------------------------
# public: calc_trend_features
# --------------------------------------------
def calc_trend_features(ohlcv: pd.DataFrame) -> Dict[str, float]:
    """
    screener.py から import される前提の関数。
    入力 ohlcv は最低限 ['Open','High','Low','Close','Volume'] を含む想定。
    """
    if ohlcv is None or len(ohlcv) < 60:
        return {
            "close": np.nan,
            "ma20": np.nan,
            "ma50": np.nan,
            "ma200": np.nan,
            "ma_stack": 0.0,
            "trend_up": 0.0,
            "pullback_ok": 0.0,
            "rsi14": np.nan,
            "atr14": np.nan,
            "atrp": np.nan,
            "ret_5d": np.nan,
            "ret_20d": np.nan,
            "vol20": np.nan,
            "adv20": np.nan,
            "hi20": np.nan,
            "lo20": np.nan,
        }

    df = ohlcv.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"OHLCV missing column: {col}")

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)

    ma20 = _sma(close, 20)
    ma50 = _sma(close, 50)
    ma200 = _sma(close, 200)

    rsi14 = _rsi(close, 14)
    atr14 = _atr(high, low, close, 14)

    # returns
    ret_5d = close.pct_change(5)
    ret_20d = close.pct_change(20)

    # highs/lows
    hi20 = high.rolling(20, min_periods=20).max()
    lo20 = low.rolling(20, min_periods=20).min()

    # volume stats
    vol20 = vol.rolling(20, min_periods=20).mean()
    adv20 = (close * vol).rolling(20, min_periods=20).mean()

    c = _last(close)
    m20 = _last(ma20)
    m50 = _last(ma50)
    m200 = _last(ma200)

    # trend definition: MA structure up
    ma_stack = 1.0 if (m20 > m50 > m200) else 0.0
    trend_up = 1.0 if (c > m50 and ma_stack > 0.5) else 0.0

    # pullback: price near MA20-50 zone (押し目)
    # zone: between MA20*(0.985) and MA50*(1.015) roughly
    pullback_ok = 0.0
    if np.isfinite(m20) and np.isfinite(m50) and np.isfinite(c):
        low_zone = min(m20, m50) * 0.985
        high_zone = max(m20, m50) * 1.015
        if low_zone <= c <= high_zone:
            pullback_ok = 1.0

    atrv = _last(atr14)
    atrp = _safe_pct(atrv, c) * 100.0 if np.isfinite(atrv) and np.isfinite(c) else np.nan

    return {
        "close": c,
        "ma20": m20,
        "ma50": m50,
        "ma200": m200,
        "ma_stack": float(ma_stack),
        "trend_up": float(trend_up),
        "pullback_ok": float(pullback_ok),
        "rsi14": _last(rsi14),
        "atr14": atrv,
        "atrp": atrp,
        "ret_5d": _last(ret_5d),
        "ret_20d": _last(ret_20d),
        "vol20": _last(vol20),
        "adv20": _last(adv20),
        "hi20": _last(hi20),
        "lo20": _last(lo20),
    }


# --------------------------------------------
# optional: unified compute
# --------------------------------------------
def compute_features(ohlcv: pd.DataFrame) -> Dict[str, float]:
    """
    他のコードが compute_features を期待しても落ちないように用意。
    """
    return calc_trend_features(ohlcv)