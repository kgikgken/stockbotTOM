from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.scoring import _add_indicators, _last_val


# ============================================================
# 20日ボラ計算（内部版）
# ============================================================

def _calc_vola20(hist: pd.DataFrame) -> float:
    if hist is None or len(hist) < 21:
        return np.nan
    close = hist["Close"].astype(float)
    ret = close.pct_change(fill_method=None)
    try:
        return float(ret.rolling(20).std().iloc[-1])
    except Exception:
        return np.nan


# ============================================================
# ボラティリティ → ベースTP/SL
# ============================================================

def _base_tp_sl_from_vola(vola: float) -> Tuple[float, float]:
    if not np.isfinite(vola):
        return 0.09, -0.045

    v = float(abs(vola))

    if v < 0.015:
        tp = 0.06
        sl = -0.03
    elif v < 0.03:
        tp = 0.09
        sl = -0.045
    elif v < 0.06:
        tp = 0.12
        sl = -0.06
    else:
        tp = 0.16
        sl = -0.085

    return tp, sl


# ============================================================
# 地合い補正
# ============================================================

def _market_multipliers(mkt_score: int) -> Tuple[float, float]:
    s = int(mkt_score)

    if s >= 75:
        return 1.20, 0.90
    if s >= 65:
        return 1.12, 0.95
    if s >= 55:
        return 1.05, 1.00
    if s >= 45:
        return 0.95, 1.05
    if s >= 35:
        return 0.90, 1.10
    return 0.85, 1.15


# ============================================================
# 波の強さ（押し目完成度）
# ============================================================

def _wave_strength(df: pd.DataFrame) -> Tuple[float, float]:
    rsi = _last_val(df.get("rsi14"))
    off = _last_val(df.get("off_high_pct"))
    days = _last_val(df.get("days_since_high60"))
    slope = _last_val(df.get("trend_slope20"))
    shadow = _last_val(df.get("lower_shadow_ratio"))

    tp_mult = 1.0
    risk_mult = 1.0

    # RSI
    if np.isfinite(rsi):
        if 32 <= rsi <= 45:
            tp_mult += 0.18
            risk_mult -= 0.05
        elif 25 <= rsi < 32 or 45 < rsi <= 55:
            tp_mult += 0.08
        elif rsi < 20 or rsi > 70:
            tp_mult -= 0.10
            risk_mult += 0.05

    # 押し幅
    if np.isfinite(off):
        if -18 <= off <= -7:
            tp_mult += 0.15
            risk_mult -= 0.05
        elif -25 <= off < -18 or -7 < off <= 0:
            tp_mult += 0.05
        else:
            tp_mult -= 0.10
            risk_mult += 0.05

    # 日柄
    if np.isfinite(days):
        if 3 <= days <= 12:
            tp_mult += 0.08
        elif 1 <= days < 3 or 12 < days <= 25:
            tp_mult += 0.03
        elif days > 30:
            tp_mult -= 0.05

    # トレンド方向
    if np.isfinite(slope):
        if slope >= 0.006:
            tp_mult += 0.10
            risk_mult -= 0.05
        elif slope >= 0.0:
            tp_mult += 0.03
        elif slope < -0.004:
            tp_mult -= 0.12
            risk_mult += 0.08

    # ヒゲ
    if np.isfinite(shadow):
        if shadow >= 0.6:
            tp_mult += 0.05
            risk_mult -= 0.03
        elif shadow <= 0.2:
            tp_mult -= 0.05

    tp_mult = float(np.clip(tp_mult, 0.7, 1.6))
    risk_mult = float(np.clip(risk_mult, 0.85, 1.25))

    return tp_mult, risk_mult


# ============================================================
# 流動性補正
# ============================================================

def _liquidity_risk_mult(df: pd.DataFrame) -> float:
    turnover20 = _last_val(df.get("turnover_avg20"))

    if not np.isfinite(turnover20):
        return 1.05

    t = float(turnover20)

    if t >= 5e9:
        return 0.95
    if t >= 1e9:
        return 1.00
    if t >= 3e8:
        return 1.05
    return 1.12


# ============================================================
# 公開API
# ============================================================

def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int) -> Dict[str, float]:
    if hist is None or len(hist) < 40:
        return {"tp_pct": 0.08, "sl_pct": -0.06, "rr": 1.33}

    df = _add_indicators(hist)

    vola20 = _calc_vola20(hist)
    base_tp, base_sl = _base_tp_sl_from_vola(vola20)

    mkt_tp_mult, mkt_sl_mult = _market_multipliers(mkt_score)

    wave_tp_mult, wave_risk_mult = _wave_strength(df)

    liq_mult = _liquidity_risk_mult(df)

    tp_pct = base_tp * mkt_tp_mult * wave_tp_mult
    sl_pct = base_sl * mkt_sl_mult * wave_risk_mult * liq_mult

    tp_pct = float(np.clip(tp_pct, 0.05, 0.25))
    sl_pct = float(np.clip(sl_pct, -0.12, -0.02))

    rr = tp_pct / abs(sl_pct) if abs(sl_pct) > 1e-6 else 0.0
    rr = float(np.clip(rr, 0.5, 6.0))

    return {
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "rr": rr,
    }