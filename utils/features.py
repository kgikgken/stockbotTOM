# utils/features.py
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


# ----------------------------
# 基本ユーティリティ
# ----------------------------
def _safe(v, default=np.nan) -> float:
    try:
        x = float(v)
        if not np.isfinite(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


# ----------------------------
# トレンド・位置関係
# ----------------------------
def trend_features(df: pd.DataFrame) -> Dict[str, float]:
    c = df["Close"].astype(float)
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()

    slope20 = _safe(sma20.diff(5).iloc[-1])
    above_20 = _safe(c.iloc[-1] - sma20.iloc[-1])
    above_50 = _safe(c.iloc[-1] - sma50.iloc[-1])

    return {
        "sma20": _safe(sma20.iloc[-1]),
        "sma50": _safe(sma50.iloc[-1]),
        "slope20": slope20,
        "above_sma20": above_20,
        "above_sma50": above_50,
        "trend_up": float(
            (c.iloc[-1] > sma20.iloc[-1]) and (sma20.iloc[-1] > sma50.iloc[-1])
        ),
    }


# ----------------------------
# モメンタム / RSI
# ----------------------------
def momentum_features(df: pd.DataFrame) -> Dict[str, float]:
    c = df["Close"].astype(float)
    rsi14 = _rsi(c, 14)

    ret5 = _safe(c.iloc[-1] / c.iloc[-6] - 1.0) if len(c) >= 6 else np.nan
    ret20 = _safe(c.iloc[-1] / c.iloc[-21] - 1.0) if len(c) >= 21 else np.nan

    return {
        "rsi14": _safe(rsi14.iloc[-1]),
        "ret5": ret5,
        "ret20": ret20,
    }


# ----------------------------
# ボラ・速度
# ----------------------------
def volatility_features(df: pd.DataFrame) -> Dict[str, float]:
    c = df["Close"].astype(float)
    atr14 = _atr(df, 14)
    atr = _safe(atr14.iloc[-1])
    atr_pct = _safe(atr / c.iloc[-1]) if c.iloc[-1] > 0 else np.nan

    return {
        "atr": atr,
        "atr_pct": atr_pct,
    }


# ----------------------------
# 出来高品質（押し枯れ/ブレイク）
# ----------------------------
def volume_features(df: pd.DataFrame) -> Dict[str, float]:
    vol = df["Volume"].astype(float)
    vma20 = vol.rolling(20).mean()

    last = _safe(vol.iloc[-1])
    ratio = _safe(last / vma20.iloc[-1]) if vma20.iloc[-1] > 0 else np.nan

    return {
        "vol_ratio": ratio,  # <1 押し枯れ / >1.5 ブレイク初動
    }


# ----------------------------
# ギャップ（GU）判定
# ----------------------------
def gap_features(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) < 2:
        return {"gu_flag": 0.0}

    o = _safe(df["Open"].iloc[-1])
    prev_c = _safe(df["Close"].iloc[-2])
    atr14 = _atr(df, 14)
    atr = _safe(atr14.iloc[-1])

    gu = float(o > prev_c + atr)
    return {"gu_flag": gu}


# ----------------------------
# Pwin 代理（0〜1）
# ----------------------------
def estimate_pwin(
    trend: Dict[str, float],
    mom: Dict[str, float],
    vol: Dict[str, float],
    volu: Dict[str, float],
    sector_rank: Optional[int] = None,
) -> float:
    """
    代理Pwin：
    - トレンド優位
    - RSI過熱回避
    - 押し枯れ or 健全出来高
    - セクター順位は微加点のみ（選定理由にしない）
    """
    score = 0.0
    w_sum = 0.0

    # Trend
    score += 0.35 * float(trend.get("trend_up", 0.0))
    w_sum += 0.35

    # RSI（40〜62を最良帯）
    rsi = mom.get("rsi14", np.nan)
    if np.isfinite(rsi):
        if 40 <= rsi <= 62:
            score += 0.25
        elif 35 <= rsi <= 70:
            score += 0.15
        w_sum += 0.25

    # 出来高
    vr = volu.get("vol_ratio", np.nan)
    if np.isfinite(vr):
        if vr < 1.0:
            score += 0.15  # 押し枯れ
        elif vr <= 1.8:
            score += 0.10
        w_sum += 0.15

    # ボラ（低すぎ/高すぎ回避）
    atr_pct = vol.get("atr_pct", np.nan)
    if np.isfinite(atr_pct):
        if 0.015 <= atr_pct <= 0.06:
            score += 0.15
        w_sum += 0.15

    # セクター順位（微加点）
    if sector_rank is not None and sector_rank <= 5:
        score += 0.05
        w_sum += 0.05

    if w_sum <= 0:
        return 0.0

    return float(min(max(score / w_sum, 0.0), 1.0))


# ----------------------------
# まとめて特徴量生成
# ----------------------------
def build_features(
    df: pd.DataFrame,
    *,
    sector_rank: Optional[int] = None,
) -> Dict[str, float]:
    trend = trend_features(df)
    mom = momentum_features(df)
    vol = volatility_features(df)
    volu = volume_features(df)
    gap = gap_features(df)

    pwin = estimate_pwin(trend, mom, vol, volu, sector_rank)

    out = {}
    out.update(trend)
    out.update(mom)
    out.update(vol)
    out.update(volu)
    out.update(gap)
    out["pwin"] = pwin
    return out