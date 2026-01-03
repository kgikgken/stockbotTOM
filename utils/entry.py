# utils/entry.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


# ----------------------------
# ユーティリティ
# ----------------------------
def _safe(v, default=np.nan) -> float:
    try:
        x = float(v)
        if not np.isfinite(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


# ----------------------------
# INゾーン計算
# ----------------------------
def calc_in_zone(
    df: pd.DataFrame,
    *,
    setup_type: str,
    atr: float,
) -> Dict[str, float]:
    """
    setup_type:
      - 'A'：押し目（SMA20基準 ±0.5ATR）
      - 'B'：ブレイク（HH20基準 ±0.3ATR）
    """
    c = df["Close"].astype(float)

    if setup_type == "A":
        sma20 = c.rolling(20).mean().iloc[-1]
        center = _safe(sma20)
        band = 0.5 * atr
    elif setup_type == "B":
        hh20 = c.rolling(20).max().iloc[-1]
        center = _safe(hh20)
        band = 0.3 * atr
    else:
        raise ValueError(f"unknown setup_type: {setup_type}")

    return {
        "in_center": center,
        "in_low": center - band,
        "in_high": center + band,
        "band": band,
    }


# ----------------------------
# ギャップ（GU）判定
# ----------------------------
def is_gap_up(
    df: pd.DataFrame,
    *,
    atr: float,
) -> bool:
    if len(df) < 2:
        return False
    o = _safe(df["Open"].iloc[-1])
    prev_c = _safe(df["Close"].iloc[-2])
    return bool(o > prev_c + atr)


# ----------------------------
# 乖離率
# ----------------------------
def calc_deviation(
    *,
    price: float,
    in_center: float,
    atr: float,
) -> float:
    if atr <= 0:
        return np.nan
    return abs(price - in_center) / atr


# ----------------------------
# 行動判定（完全機械化）
# ----------------------------
def decide_action(
    *,
    price: float,
    in_low: float,
    in_high: float,
    in_center: float,
    atr: float,
    gu_flag: bool,
) -> Dict[str, object]:
    """
    Action:
      - EXEC_NOW   : 帯の中 & GUなし & 乖離<=0.4
      - LIMIT_WAIT : 帯外だが乖離<=0.8 & GUなし
      - WATCH_ONLY : GUあり or 乖離>0.8
    """
    deviation = calc_deviation(
        price=price,
        in_center=in_center,
        atr=atr,
    )

    if gu_flag:
        return {
            "action": "WATCH_ONLY",
            "reason": "GU",
            "deviation": deviation,
        }

    # 帯の中
    if in_low <= price <= in_high:
        if deviation <= 0.4:
            return {
                "action": "EXEC_NOW",
                "reason": "IN_ZONE",
                "deviation": deviation,
            }
        else:
            return {
                "action": "LIMIT_WAIT",
                "reason": "IN_ZONE_BUT_FAR",
                "deviation": deviation,
            }

    # 帯の外
    if deviation <= 0.8:
        return {
            "action": "LIMIT_WAIT",
            "reason": "WAIT_PULLBACK",
            "deviation": deviation,
        }

    return {
        "action": "WATCH_ONLY",
        "reason": "TOO_FAR",
        "deviation": deviation,
    }


# ----------------------------
# まとめ処理
# ----------------------------
def build_entry_decision(
    df: pd.DataFrame,
    *,
    setup_type: str,
    atr: float,
) -> Dict[str, object]:
    """
    出力：
      - in_center / in_low / in_high
      - deviation
      - action / reason
      - gu_flag
    """
    zone = calc_in_zone(df, setup_type=setup_type, atr=atr)
    price = _safe(df["Close"].iloc[-1])
    gu = is_gap_up(df, atr=atr)

    decision = decide_action(
        price=price,
        in_low=zone["in_low"],
        in_high=zone["in_high"],
        in_center=zone["in_center"],
        atr=atr,
        gu_flag=gu,
    )

    out = {}
    out.update(zone)
    out.update(decision)
    out["price"] = price
    out["gu_flag"] = gu
    return out