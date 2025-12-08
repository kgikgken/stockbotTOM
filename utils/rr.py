from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from utils.scoring import _add_indicators, calc_vola20


# ============================================================
# 内部ヘルパー
# ============================================================
def _last(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return np.nan
    v = series.iloc[-1]
    try:
        return float(v)
    except Exception:
        return np.nan


# ============================================================
# base TP / SL（ボラティリティベース）
# ============================================================
def base_tp_sl(vola20: float) -> Tuple[float, float]:
    """
    vola20 から TP/SL のベース値を決める
    戻り値は (tp_pct, sl_pct<0)
    """
    v = abs(vola20) if np.isfinite(vola20) else 0.03

    # 低ボラ：6% / -3%
    if v < 0.015:
        tp = 0.06
        sl = -0.03
    # 中ボラ：8% / -4%
    elif v < 0.03:
        tp = 0.08
        sl = -0.04
    # やや高ボラ：10% / -5%
    elif v < 0.05:
        tp = 0.10
        sl = -0.05
    # 高ボラ：12% / -6%
    else:
        tp = 0.12
        sl = -0.06

    return float(tp), float(sl)


# ============================================================
# 地合い補正
# ============================================================
def mkt_adjust(mkt_score: int) -> float:
    """
    地合いスコア（0〜100）から上振れ・下振れ補正を返す。
    例:
      0.2 → 強気
      0.0 → 中立
      -0.1 → 弱気
    """
    s = int(mkt_score)

    if s >= 75:
        return 0.25
    if s >= 65:
        return 0.15
    if s >= 55:
        return 0.05
    if s <= 40:
        return -0.10
    return 0.0


# ============================================================
# 押し目完成度（RSI / 押し幅 / 日柄 / ヒゲ）
# ============================================================
def pullback_coef(df: pd.DataFrame) -> float:
    """
    押し目の完成度を -0.3〜+0.3 で返す。
    プラスほど「理想押し目」、マイナスほど「汚い押し」。
    """
    rsi = _last(df["rsi14"])
    off = _last(df["off_high_pct"])
    days = _last(df["days_since_high60"])
    shadow = _last(df["lower_shadow_ratio"])

    coef = 0.0

    # RSI
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            coef += 0.15
        elif 20 <= rsi < 30 or 45 < rsi <= 55:
            coef += 0.07
        elif rsi < 20 or rsi > 70:
            coef -= 0.10

    # 高値からの押し幅
    if np.isfinite(off):
        if -12 <= off <= -5:
            coef += 0.12
        elif -20 <= off < -12:
            coef += 0.05
        elif off > 5 or off < -25:
            coef -= 0.10

    # 日柄（押しの日数）
    if np.isfinite(days):
        if 2 <= days <= 10:
            coef += 0.05
        elif 1 <= days < 2 or 10 < days <= 20:
            coef += 0.02

    # 下ヒゲ
    if np.isfinite(shadow):
        if shadow >= 0.5:
            coef += 0.05
        elif shadow >= 0.3:
            coef += 0.02

    # クリップ
    coef = float(np.clip(coef, -0.30, 0.30))
    return coef


# ============================================================
# メイン：TP / SL / RR 計算
# ============================================================
def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int) -> Tuple[float, float, float]:
    """
    hist（yfinance の history）と地合いスコアから、
    (tp_pct, sl_pct(<0), rr) を返す。

    - tp_pct: 利確目安（例: 0.08 → +8%）
    - sl_pct: 損切り目安（例: -0.04 → -4%）
    - rr:     RR（tp_pct / |sl_pct|）
    """
    if hist is None or len(hist) < 40:
        # データ不足時のデフォルト（中庸）
        tp, sl = 0.08, -0.04
        rr = tp / abs(sl)
        return tp, sl, rr

    # インジケータを付与（scoring.py と同じロジック）
    df = _add_indicators(hist)

    # ボラ → base TP/SL
    vola20 = calc_vola20(hist)
    base_tp, base_sl = base_tp_sl(vola20)

    # 地合い補正
    m_adj = mkt_adjust(mkt_score)

    # 押し目完成度
    p_coef = pullback_coef(df)

    # --- TP/SL 調整 ---
    # TP は強気方向に、SL は慎重方向に補正
    tp_pct = base_tp * (1.0 + m_adj + p_coef)
    sl_pct = base_sl * (1.0 - 0.5 * m_adj - p_coef)

    # 安全クリップ（極端な値を防ぐ）
    tp_pct = float(np.clip(tp_pct, 0.04, 0.20))
    sl_pct = float(np.clip(sl_pct, -0.09, -0.02))

    # RR計算
    if sl_pct >= 0:
        rr = 0.0
    else:
        rr = float(tp_pct / abs(sl_pct))

    return tp_pct, sl_pct, rr