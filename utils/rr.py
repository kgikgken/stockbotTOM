from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.scoring import _add_indicators, calc_vola20, _last_val


# ============================================================
# ボラティリティ → ベースTP/SL
# ============================================================

def _base_tp_sl_from_vola(vola: float) -> Tuple[float, float]:
    """
    ボラティリティ（20日標準偏差）から
    ベースの TP/SL (%表記, +/−) を決める。
    """
    if not np.isfinite(vola):
        # データ不足 ⇒ 中庸
        return 0.09, -0.045

    v = float(abs(vola))

    if v < 0.015:
        # 超低ボラ：リターンも小さいが騙しも少ない
        tp = 0.06
        sl = -0.03
    elif v < 0.03:
        # 低〜中ボラ：理想ゾーン
        tp = 0.09
        sl = -0.045
    elif v < 0.06:
        # やや高ボラ：取りに行くがリスクも増える
        tp = 0.12
        sl = -0.06
    else:
        # 超高ボラ：TPは伸ばすがSLも厚め
        tp = 0.16
        sl = -0.085

    return tp, sl


# ============================================================
# 地合い補正
# ============================================================

def _market_multipliers(mkt_score: int) -> Tuple[float, float]:
    """
    地合いスコアから TP/SL の倍率を返す。
    tp_mult > 1 ならTPを伸ばす、sl_mult < 1 なら損切りを浅く。
    """
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
# 波の強さ（押し目完成度）を評価
# ============================================================

def _wave_strength(df: pd.DataFrame) -> Tuple[float, float]:
    """
    押し目の完成度から
      wave_tp_mult : TP側にかける倍率
      wave_risk_mult : SL側にかける倍率
    を返す。
    """
    rsi = _last_val(df.get("rsi14"))
    off = _last_val(df.get("off_high_pct"))
    days = _last_val(df.get("days_since_high60"))
    slope = _last_val(df.get("trend_slope20"))
    shadow = _last_val(df.get("lower_shadow_ratio"))

    tp_mult = 1.0
    risk_mult = 1.0

    # --- RSI ---
    if np.isfinite(rsi):
        if 32 <= rsi <= 45:
            tp_mult += 0.18   # きれいな押し目
            risk_mult -= 0.05
        elif 25 <= rsi < 32 or 45 < rsi <= 55:
            tp_mult += 0.08
        elif rsi < 20 or rsi > 70:
            tp_mult -= 0.10
            risk_mult += 0.05

    # --- 高値からの押し幅 ---
    if np.isfinite(off):
        if -18 <= off <= -7:
            tp_mult += 0.15
            risk_mult -= 0.05
        elif -25 <= off < -18 or -7 < off <= 0:
            tp_mult += 0.05
        elif off > 0 or off < -30:
            tp_mult -= 0.10
            risk_mult += 0.05

    # --- 日柄 ---
    if np.isfinite(days):
        if 3 <= days <= 12:
            tp_mult += 0.08
        elif 1 <= days < 3 or 12 < days <= 25:
            tp_mult += 0.03
        elif days > 30:
            tp_mult -= 0.05

    # --- トレンド方向（20MAの傾き） ---
    if np.isfinite(slope):
        if slope >= 0.006:
            tp_mult += 0.10
            risk_mult -= 0.05
        elif slope >= 0.0:
            tp_mult += 0.03
        elif slope < -0.004:
            tp_mult -= 0.12
            risk_mult += 0.08

    # --- ヒゲ ---
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
# 流動性によるリスク補正
# ============================================================

def _liquidity_risk_mult(df: pd.DataFrame) -> float:
    """
    出来高・売買代金が薄いほど SL を少し広げる。
    """
    turnover20 = _last_val(df.get("turnover_avg20"))

    if not np.isfinite(turnover20):
        return 1.05

    t = float(turnover20)

    if t >= 5e9:
        return 0.95   # 超高流動性 → 少しだけタイトでOK
    if t >= 1e9:
        return 1.00   # 十分
    if t >= 3e8:
        return 1.05   # やや薄い
    return 1.12       # 薄商い → 余裕を持ったSL


# ============================================================
# 公開API：TP/SL/RR の計算
# ============================================================

def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int) -> Dict[str, float]:
    """
    1銘柄分のヒストリカルデータから
      - tp_pct : 利確目安（+◯％）
      - sl_pct : 損切り目安（−◯％）
      - rr     : 想定RR = TP幅 / |SL幅|
    を返す。
    """
    if hist is None or len(hist) < 40:
        # データ不足時は無難な設定で返す（RRは低め）
        return {"tp_pct": 0.08, "sl_pct": -0.06, "rr": 1.33}

    # インジケータ付与（scoring側と完全同一ロジック）
    df = _add_indicators(hist)

    # ボラティリティ
    vola20 = calc_vola20(hist)
    base_tp, base_sl = _base_tp_sl_from_vola(vola20)

    # 地合い補正
    mkt_tp_mult, mkt_sl_mult = _market_multipliers(mkt_score)

    # 波の完成度
    wave_tp_mult, wave_risk_mult = _wave_strength(df)

    # 流動性リスク
    liq_mult = _liquidity_risk_mult(df)

    # ---- 合成 ----
    tp_pct = base_tp * mkt_tp_mult * wave_tp_mult
    sl_pct = base_sl * mkt_sl_mult * wave_risk_mult * liq_mult

    # クリップ（やりすぎ防止）
    tp_pct = float(np.clip(tp_pct, 0.05, 0.25))
    sl_pct = float(np.clip(sl_pct, -0.12, -0.02))

    rr = tp_pct / abs(sl_pct) if abs(sl_pct) > 1e-6 else 0.0
    rr = float(np.clip(rr, 0.5, 6.0))

    return {
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "rr": rr,
    }