from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from utils.util import clamp


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return float("nan")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    v = float(tr.rolling(period).mean().iloc[-1])
    return v if np.isfinite(v) else float("nan")


def add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()
    df["ma10"] = close.rolling(10).mean()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ATR
    df["atr14"] = np.nan
    v = _atr(df, 14)
    if np.isfinite(v):
        df.loc[df.index[-1], "atr14"] = v

    # 売買代金(概算): Close * Volume
    df["turnover"] = close * vol
    df["adv20"] = df["turnover"].rolling(20).mean()

    # 20日リターン（相関用）
    df["ret1"] = close.pct_change(fill_method=None)

    return df


def calc_liquidity_flags(df: pd.DataFrame, adv_min: float) -> Tuple[bool, float]:
    adv = float(df["adv20"].iloc[-1]) if "adv20" in df.columns and len(df) >= 20 else float("nan")
    ok = bool(np.isfinite(adv) and adv >= adv_min)
    return ok, (adv if np.isfinite(adv) else 0.0)


def calc_atr_pct(df: pd.DataFrame) -> float:
    close = float(df["Close"].iloc[-1])
    atr = float(df["atr14"].iloc[-1]) if "atr14" in df.columns else float("nan")
    if not (np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0):
        return float("nan")
    return float((atr / close) * 100.0)


def estimate_pwin(feature: Dict) -> float:
    """
    代理Pwin：0.20〜0.62 の範囲に収める（過剰に盛らない）
    """
    trend = clamp(feature.get("trend", 0.0), 0.0, 1.0)
    pullback_quality = clamp(feature.get("pullback_quality", 0.0), 0.0, 1.0)
    volume = clamp(feature.get("volume_quality", 0.0), 0.0, 1.0)
    liquidity = clamp(feature.get("liquidity", 0.0), 0.0, 1.0)
    gap_risk = clamp(feature.get("gap_risk", 0.0), 0.0, 1.0)  # 1が安全

    raw = (
        0.28
        + 0.16 * trend
        + 0.10 * pullback_quality
        + 0.06 * volume
        + 0.04 * liquidity
        + 0.08 * gap_risk
    )
    return float(clamp(raw, 0.20, 0.62))


def regime_multiplier(mkt_score: int, delta3d: int, macro_caution: bool) -> float:
    mult = 1.0
    if mkt_score >= 70 and delta3d >= 0:
        mult *= 1.06
    elif mkt_score >= 60 and delta3d >= 0:
        mult *= 1.03

    if delta3d <= -5:
        mult *= 0.78
    elif delta3d <= -2:
        mult *= 0.90

    if mkt_score < 50:
        mult *= 0.90

    if macro_caution:
        mult *= 0.75

    return float(clamp(mult, 0.55, 1.10))