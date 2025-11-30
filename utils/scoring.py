import numpy as np
import pandas as pd

# =============
# 安全補助
# =============
def safe(x):
    """NaN 保護。数字でなければ None を返す"""
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except:
        return None


# =============
# スコア計算
# =============
def score_stock(df: pd.DataFrame):
    """
    Aランク / Bランク判定用の総合スコア（0–100）
    NaN が一つでも混ざると None を返す（安全フィルター）
    """

    # --- 必須カラムの存在チェック ---
    need = ["Close", "Open", "High", "Low", "Volume"]
    if any(col not in df.columns for col in need):
        return None

    # --- 指標計算に必要な終値 ---
    close = safe(df["Close"].iloc[-1])
    if close is None:
        return None

    # --- 移動平均 ---
    ma20 = safe(df["Close"].rolling(20).mean().iloc[-1])
    ma50 = safe(df["Close"].rolling(50).mean().iloc[-1])

    # --- RSI ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean().iloc[-1]
    avg_loss = loss.rolling(14).mean().iloc[-1]

    if avg_loss == 0 or avg_gain is None or avg_loss is None:
        rsi = None
    else:
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

    # 全部 safe 化
    rsi = safe(rsi)

    # --- ボラ20 ---
    vola20 = safe(df["Close"].pct_change().rolling(20).std().iloc[-1])

    # --- どれか一個でも NaN → スキップ ---
    if any(v is None for v in [ma20, ma50, rsi, vola20]):
        return None

    # ===== スコアリング =====
    score = 0

    # トレンド
    if close > ma20:
        score += 10
    if close > ma50:
        score += 10

    # RSI（押し目評価）
    if 30 <= rsi <= 45:
        score += 15
    elif 20 <= rsi < 30 or 45 < rsi <= 55:
        score += 8

    # ボラティリティ
    if vola20 < 0.02:
        score += 5
    elif vola20 < 0.06:
        score += 3

    # 上限調整
    score = int(np.clip(score, 0, 100))

    return score