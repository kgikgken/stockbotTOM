from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================
# ACDEスコア（Qualityレイヤー）
# ------------------------------------------------------------
# A: Acceleration（勢いの変化）
# C: Consistency（安定した上昇/下落）
# D: Direction（方向性）
# E: Efficiency（効率よく上げているか）
#
# 「素の銘柄力×直近の形」で、
# セットアップ前の"基礎スコア"を算出する。
# ============================================================


def score_stock(hist: pd.DataFrame) -> float:
    """
    ACDEスコアを 0〜100 で返す。
    相対評価のため、単独でも意味はあるが、
    本質は「Universe内での上下差」。
    """
    try:
        close = hist["Close"].astype(float)
    except Exception:
        return 50.0

    if len(close) < 40:
        return 50.0

    # ------------------------------------
    # 日次リターン
    # ------------------------------------
    ret = close.pct_change(fill_method=None)
    if ret.isna().all():
        return 50.0

    # ------------------------------------
    # A: Acceleration（勢いの変化）
    # 近20日の角度 vs 過去20日の角度
    # ------------------------------------
    try:
        slope_recent = np.polyfit(range(20), close[-20:], 1)[0]
        slope_prev = np.polyfit(range(20), close[-40:-20], 1)[0]
        accel = slope_recent - slope_prev
    except Exception:
        accel = 0.0

    A = np.clip(accel * 5000.0, -10.0, 20.0)

    # ------------------------------------
    # C: Consistency（安定性）
    # 20日間の勝率（陽線比率）
    # ------------------------------------
    try:
        wins = (ret[-20:] > 0).sum()
        C = (wins / 20.0) * 20.0  # 最大20点
    except Exception:
        C = 10.0

    # ------------------------------------
    # D: Direction（方向性）
    # 10日 vs 30日移動平均
    # ------------------------------------
    ma10 = float(close.rolling(10).mean().iloc[-1])
    ma30 = float(close.rolling(30).mean().iloc[-1])

    if ma10 > ma30:
        D = 20.0 * ((ma10 / (ma30 + 1e-9)) - 1.0) * 15.0
        D = np.clip(D, 0.0, 25.0)
    else:
        D = 0.0

    # ------------------------------------
    # E: Efficiency（効率）
    # 上昇率 / ボラ
    # ------------------------------------
    try:
        chg = close.iloc[-1] / close.iloc[-20] - 1.0
        vol = ret[-20:].std()
        if vol > 0:
            E = (chg / vol) * 10.0
        else:
            E = 0.0
    except Exception:
        E = 0.0

    E = np.clip(E, -5.0, 25.0)

    # ------------------------------------
    # 合計
    # ------------------------------------
    score = A + C + D + E

    # 基本範囲へ
    return float(np.clip(score, 0.0, 100.0))