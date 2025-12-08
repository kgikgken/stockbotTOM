from __future__ import annotations
import numpy as np
import pandas as pd

from utils.scoring import _add_indicators


# ============================================================
# 1. ボラ基準の TP / SL（素）
# ============================================================
def base_tp_sl(vola20: float) -> tuple[float, float]:
    """
    ボラティリティから基礎TP/SL%を返す
    戻り値: (tp_pct, sl_pct) 例: (0.08, -0.04)
    """
    v = float(vola20) if np.isfinite(vola20) else 0.03

    # 低ボラ
    if v < 0.02:
        return 0.06, -0.03

    # 中間ボラ
    if v < 0.035:
        return 0.08, -0.04

    # 高ボラ
    return 0.12, -0.06


# ============================================================
# 2. 地合い補正
# ============================================================
def mkt_adjust(tp: float, sl: float, mkt_score: int) -> tuple[float, float]:
    """
    地合いで TP/SL を微調整
    """
    s = int(mkt_score)

    # 強地合い ← TP伸ばす
    if s >= 70:
        tp *= 1.10

    # 弱地合い ← SL浅く
    elif s < 45:
        tp *= 0.90
        sl = max(sl, -0.03)

    # clip
    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))
    return tp, sl


# ============================================================
# 3. 押し目完成度（係数）
# ============================================================
def pullback_coef(df: pd.DataFrame) -> float:
    """
    RSI + off_high + ヒゲ から押し目完成度を推定（0.8〜1.2倍）
    """
    def lv(x):
        return float(x) if np.isfinite(x) else np.nan

    rsi = lv(df["rsi14"].iloc[-1])
    off = lv(df["off_high_pct"].iloc[-1])
    shadow = lv(df["lower_shadow_ratio"].iloc[-1])

    c = 1.0

    # RSI（30〜45が最高）
    if 30 <= rsi <= 45:
        c += 0.10
    elif 45 < rsi <= 55:
        c += 0.05
    elif rsi <= 25:
        c -= 0.05

    # 高値からの下落 -5〜-18%が理想押し目
    if -18 <= off <= -5:
        c += 0.10
    elif -25 <= off < -18:
        c += 0.05
    elif off >= 0:
        c -= 0.10

    # 下ヒゲ → 買い圧（反転兆候）
    if shadow >= 0.5:
        c += 0.05
    elif shadow >= 0.3:
        c += 0.02

    return float(np.clip(c, 0.8, 1.2))


# ============================================================
# 4. RR計算
# ============================================================
def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int) -> tuple[float, float, float]:
    """
    戻り値:
      tp_pct (+: 利確%),
      sl_pct (-: 損切%),
      rr (TP幅 / |SL幅|)
    """
    if hist is None or len(hist) < 40:
        return 0.06, -0.03, 2.0

    # 指標計算
    df = _add_indicators(hist)

    # vola 取得
    vola20 = float(df["vola20"].iloc[-1]) if np.isfinite(df["vola20"].iloc[-1]) else 0.03

    # 1) ボラベースの素
    tp, sl = base_tp_sl(vola20)

    # 2) 押し目係数
    coef = pullback_coef(df)
    tp *= coef
    sl *= coef

    # 3) 地合い補正
    tp, sl = mkt_adjust(tp, sl, mkt_score)

    # RR
    rr = float(tp / abs(sl)) if sl != 0 else 2.0

    # 最終clip
    tp = float(np.clip(tp, 0.04, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))
    return tp, sl, rr