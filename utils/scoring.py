import numpy as np
import pandas as pd

# ============================================================
# 指標計算補助（NaN安全）
# ============================================================

def nz(x, default=np.nan):
    """NaN → default"""
    try:
        if x is None:
            return default
        if np.isnan(float(x)):
            return default
        return float(x)
    except:
        return default


# ============================================================
# Aランク / Bランク スクリーニング用スコアリング
# ============================================================

def score_stock(df: pd.DataFrame) -> float:
    """
    df: yfinance OHLCV + Close/MA/RSI が計算済み
    戻り値: 0〜100（NaNの場合は -1 を返して落とす）
    """

    try:
        last = df.iloc[-1]
    except:
        return -1

    close = nz(last.get("Close"))
    ma20 = nz(last.get("ma20"))
    ma50 = nz(last.get("ma50"))
    rsi = nz(last.get("rsi14"))
    shadow = nz(last.get("lower_shadow_ratio"))
    vola = nz(last.get("vola20"))
    off = nz(last.get("off_high_pct"))
    days = nz(last.get("days_since_high60"))

    if any(np.isnan([close, ma20, ma50, rsi])):
        return -1  # 指標欠損はスキップ

    score = 0

    # ----------------------------------------
    # ① トレンド強度（最大30点）
    # ----------------------------------------
    # MA配置
    if close > ma20 > ma50:
        score += 18
    elif close > ma20:
        score += 10
    elif ma20 > ma50:
        score += 6

    # 高値からの距離（押し目判定）
    if not np.isnan(off):
        if -8 <= off <= -4:
            score += 12
        elif -15 <= off < -8:
            score += 7

    # ----------------------------------------
    # ② RSI押しの質（最大20点）
    # ----------------------------------------
    if 30 <= rsi <= 45:
        score += 12
    elif 20 <= rsi < 30 or 45 < rsi <= 55:
        score += 6

    # ----------------------------------------
    # ③ 反転サイン（最大10点）
    # ----------------------------------------
    if shadow >= 0.5:
        score += 6
    elif shadow >= 0.3:
        score += 3

    # ----------------------------------------
    # ④ 流動性 & ボラ（最大20点）
    # ----------------------------------------
    if not np.isnan(vola):
        if vola < 0.02:
            score += 12
        elif vola < 0.05:
            score += 6

    # ----------------------------------------
    # ⑤ 日柄調整（最大10点）
    # ----------------------------------------
    if 2 <= days <= 12:
        score += 8
    elif 1 <= days < 2 or 12 < days <= 25:
        score += 4

    return float(score)


# ============================================================
# Core分類（Aランク / Bランク）
# ============================================================

def classify_core(score: float):
    """
    A: 75〜100
    B: 60〜74
    C: 0〜59（返さない）
    """
    if score >= 75:
        return "A"
    if score >= 60:
        return "B"
    return None
