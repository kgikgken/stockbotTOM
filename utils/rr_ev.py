from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from utils.util import safe_float, clamp
from utils.features import atr, add_indicators


def compute_rr_targets(df_raw: pd.DataFrame, setup: str, entry: Dict) -> Dict:
    """
    STOP/TP1/TP2（構造＋ATR）
    RR = (TP2-IN)/(IN-STOP)
    """
    df = add_indicators(df_raw)
    close = df["Close"].astype(float)

    in_center = safe_float(entry.get("in_center"))
    in_low = safe_float(entry.get("in_low"))
    a = safe_float(entry.get("atr"))
    if not np.isfinite(a) or a <= 0:
        a = atr(df_raw, 14)
    if not np.isfinite(a) or a <= 0:
        a = max(safe_float(close.iloc[-1]) * 0.01, 1.0)

    # STOP
    lookback = 12
    swing_low = float(df_raw["Low"].astype(float).tail(lookback).min()) if len(df_raw) >= lookback else float(df_raw["Low"].astype(float).min())

    if setup == "A":
        # STOP = IN_low - 0.7ATR（≒ center - 1.2ATR）
        stop = in_low - 0.7 * a
    elif setup == "B":
        # STOP = BreakLine - 1.0ATR（center を BreakLine とみなす）
        stop = in_center - 1.0 * a
    else:
        stop = in_center - 1.2 * a

    # 構造安値が近いならさらに下へ
    stop = min(stop, swing_low - 0.2 * a)

    risk = in_center - stop
    if risk <= 0:
        return {"rr": 0.0, "stop": float(stop), "tp1": float(in_center), "tp2": float(in_center), "r": float(risk)}

    # TP（2段階）
    tp1 = in_center + 1.5 * risk
    tp2 = in_center + 3.0 * risk

    rr = (tp2 - in_center) / risk

    return {
        "stop": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rr": float(rr),
        "risk_yen": float(risk),
    }


def pwin_proxy(
    df_raw: pd.DataFrame,
    setup: str,
    rs20: float,
    sector_rank: int | None,
    adv20: float,
    gu: bool,
) -> float:
    """
    ログ無しでの “代理Pwin”。
    0〜1（過剰に高くしない。現実寄せ）
    """
    df = add_indicators(df_raw)
    close = df["Close"].astype(float)
    c = safe_float(close.iloc[-1])

    ma20 = safe_float(df["ma20"].iloc[-1])
    ma50 = safe_float(df["ma50"].iloc[-1])
    slope5 = safe_float(df["ma20_slope5"].iloc[-1])
    rsi = safe_float(df["rsi14"].iloc[-1])

    score = 0.0

    # TrendStrength
    if np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma50):
        if c > ma20 > ma50:
            score += 0.22
        elif c > ma20:
            score += 0.12

    if np.isfinite(slope5):
        score += clamp(slope5 * 12.0, -0.08, 0.12)

    # RSI（過熱は減点）
    if np.isfinite(rsi):
        if 40 <= rsi <= 62:
            score += 0.12
        elif rsi >= 70 or rsi <= 35:
            score -= 0.10

    # RS（強いほど加点）
    if np.isfinite(rs20):
        score += clamp(rs20 / 40.0, -0.08, 0.12)

    # SectorRank（上位ほど微加点：ただし理由にしない）
    if sector_rank is not None and sector_rank > 0:
        # 1位=+0.08, 5位=+0.04, 10位=+0.02, それ以降ほぼ0
        score += clamp(0.10 * (1.0 / (1.0 + (sector_rank - 1) / 4.0)), 0.0, 0.08)

    # Liquidity（ADV）
    if np.isfinite(adv20) and adv20 > 0:
        if adv20 >= 1e9:
            score += 0.06
        elif adv20 >= 2e8:
            score += 0.03

    # GapRisk
    if gu:
        score -= 0.18

    # setup補正（Bは難易度高いので保守的）
    if setup == "B":
        score -= 0.05

    # ベース勝率（現実寄せ）
    base = 0.34
    p = base + score
    return float(clamp(p, 0.18, 0.55))


def ev_and_speed(rr: float, pwin: float, atr_yen: float, in_price: float, tp2: float) -> Dict:
    """
    EV(R) と ExpectedDays / Rday
    """
    if rr <= 0:
        return {"ev": -999.0, "exp_days": 999.0, "r_day": 0.0}

    ev = pwin * rr - (1.0 - pwin) * 1.0

    # ExpectedDays = (TP2-IN)/(k*ATR)
    k = 1.0
    move = tp2 - in_price
    if not np.isfinite(move) or move <= 0 or not np.isfinite(atr_yen) or atr_yen <= 0:
        exp_days = 999.0
    else:
        exp_days = float(move / (k * atr_yen))
        exp_days = clamp(exp_days, 0.5, 30.0)

    r_day = float(rr / exp_days) if exp_days > 0 else 0.0

    return {"ev": float(ev), "exp_days": float(exp_days), "r_day": float(r_day)}