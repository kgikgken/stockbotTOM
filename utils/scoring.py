from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, Tuple, Optional


SetupType = Literal["A", "B"]


def _last(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return float("nan")


def _atr14(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return float("nan")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    df["ma10"] = c.rolling(10).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()

    # 20日高値（当日含む/除くの両方）
    df["hh20"] = c.rolling(20).max()
    df["hh20_prev"] = df["hh20"].shift(1)

    # 出来高
    df["vol_ma20"] = v.rolling(20).mean()

    # 売買代金（20日平均）
    df["adv20"] = (c * v).rolling(20).mean()

    # SMA20の傾き（5日前との差分）
    df["ma20_slope5"] = df["ma20"] - df["ma20"].shift(5)

    # ATR
    df["atr14"] = _atr14(df, 14)

    return df


def universe_pass(hist: pd.DataFrame) -> Tuple[bool, float, float, float]:
    """
    Returns: (pass, close, adv20, atr_pct)
    """
    if hist is None or len(hist) < 120:
        return False, float("nan"), float("nan"), float("nan")

    df = add_indicators(hist)
    c = _last(df["Close"])
    adv20 = _last(df["adv20"])
    atr = _last(df["atr14"])

    if not (np.isfinite(c) and 200.0 <= c <= 15000.0):
        return False, c, adv20, float("nan")

    if not (np.isfinite(adv20) and adv20 >= 100_000_000.0):
        return False, c, adv20, float("nan")

    if not (np.isfinite(atr) and atr > 0):
        return False, c, adv20, float("nan")

    atr_pct = atr / c * 100.0
    if atr_pct < 1.5:
        return False, c, adv20, atr_pct

    return True, c, adv20, atr_pct


def detect_setup(hist: pd.DataFrame) -> Tuple[Optional[SetupType], str]:
    """
    Setup A: トレンド押し目
    Setup B: ブレイク（厳選）

    Returns: (setup_type, in_rank)
    """
    if hist is None or len(hist) < 120:
        return None, "様子見"

    df = add_indicators(hist)

    c = _last(df["Close"])
    ma20 = _last(df["ma20"])
    ma50 = _last(df["ma50"])
    slope5 = _last(df["ma20_slope5"])
    atr = _last(df["atr14"])

    if not (np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma50) and np.isfinite(atr) and atr > 0):
        return None, "様子見"

    # --- Setup A（最優先） ---
    # Close > SMA20 > SMA50, SMA20 slope+, CloseがSMA20に接近（<=0.8ATR）
    if (c > ma20 > ma50) and (slope5 > 0):
        dist = abs(c - ma20)
        if dist <= 0.8 * atr:
            # 強/通常の差：接近度
            if dist <= 0.4 * atr:
                return "A", "強IN"
            return "A", "通常IN"

    # --- Setup B（厳選） ---
    # Close > HH20(prev) かつ Vol >=1.5*VolMA20
    hh20_prev = _last(df["hh20_prev"])
    vol = _last(df["Volume"]) if "Volume" in df.columns else float("nan")
    vol_ma20 = _last(df["vol_ma20"])

    if np.isfinite(hh20_prev) and np.isfinite(vol) and np.isfinite(vol_ma20) and vol_ma20 > 0:
        if (c > hh20_prev) and (vol >= 1.5 * vol_ma20):
            # 抜けた日だけ（追いかけ防止）：ブレイクラインから0.6ATR以内
            if abs(c - hh20_prev) <= 0.6 * atr:
                return "B", "通常IN"

    return None, "様子見"


def in_zone_center(hist: pd.DataFrame, setup: SetupType) -> Tuple[float, float]:
    """
    Returns: (center, atr)
    A: SMA20
    B: HH20(prev)
    """
    df = add_indicators(hist)
    atr = _last(df["atr14"])
    if not (np.isfinite(atr) and atr > 0):
        atr = float("nan")

    if setup == "A":
        center = _last(df["ma20"])
    else:
        center = _last(df["hh20_prev"])
    return float(center), float(atr)


def gu_flag(hist: pd.DataFrame) -> bool:
    if hist is None or len(hist) < 2:
        return False
    df = add_indicators(hist)
    atr = _last(df["atr14"])
    if not (np.isfinite(atr) and atr > 0):
        return False
    o = float(df["Open"].astype(float).iloc[-1])
    prev_c = float(df["Close"].astype(float).iloc[-2])
    return bool(o > prev_c + 1.0 * atr)


def pwin_proxy(
    in_rank: str,
    setup: SetupType,
    sector_rank: int,
    mkt_score: int,
    delta_mkt_3d: int,
    adv20: float,
) -> float:
    """
    ログ無しの代理Pwin（最小限）
    - A > B
    - セクター上位ほど加点
    - 地合い悪化は減点
    - 流動性は下限以上なら微加点
    """
    p = 0.34
    if setup == "A":
        p += 0.04
    if in_rank == "強IN":
        p += 0.03

    # sector_rank: 1..5
    if 1 <= sector_rank <= 5:
        p += (6 - sector_rank) * 0.006  # max +0.03

    if mkt_score >= 60:
        p += 0.01
    if mkt_score < 50:
        p -= 0.01

    if delta_mkt_3d <= -5:
        p -= 0.03

    if np.isfinite(adv20) and adv20 >= 200_000_000:
        p += 0.01

    return float(np.clip(p, 0.20, 0.55))
