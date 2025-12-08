from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 地合いスコア（0〜100）
# ============================================================
def calc_market_score() -> dict:
    """
    日本株スイング向けの「地合いスコア（0〜100）」とコメントを返す。
    目的：IN/OUTのタイミング & レバレッジ判断

    評価軸：
    1. TOPIXの短期〜中期トレンド
    2. RSI（過熱・売られ過ぎ）
    3. 20日ボラ
    4. 5日騰落
    """

    try:
        data = yf.Ticker("^TOPX").history(period="120d")
        if data is None or data.empty:
            return {"score": 50, "comment": "データ不足（中立）"}
    except Exception:
        return {"score": 50, "comment": "データ取得失敗（中立）"}

    close = data["Close"].astype(float)

    # --- MA ---
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]

    # --- RSI ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # --- 20日ボラ ---
    vola20 = close.pct_change().rolling(20).std().iloc[-1]

    # --- 5日騰落 ---
    chg5 = (close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0

    score = 50.0

    # MAの並び（強さ）
    if ma5 > ma20 > ma60:
        score += 20
    elif ma5 > ma20:
        score += 15
    elif ma20 > ma60:
        score += 7
    else:
        score -= 10

    # RSI（中庸が最高）
    if 42 <= rsi <= 58:
        score += 15
    elif 35 <= rsi <= 65:
        score += 7
    else:
        score -= 8

    # ボラ（低〜適度が良い）
    if vola20 < 0.012:
        score -= 4  # ボラ小さすぎ=値幅出ない
    elif vola20 < 0.03:
        score += 8
    elif vola20 < 0.055:
        score += 2
    else:
        score -= 10

    # 5日騰落（地合いの短期強さ）
    if chg5 > 3:
        score += 7
    elif chg5 > 0:
        score += 3
    else:
        score -= 5

    score = float(np.clip(round(score), 0, 100))

    # コメント生成
    if score >= 70:
        comment = "強め（押し目＋一部ブレイク可）"
    elif score >= 60:
        comment = "やや強め（押し目メイン）"
    elif score >= 50:
        comment = "普通（押し目）"
    elif score >= 40:
        comment = "守り（小ロット）"
    else:
        comment = "弱い（最小ロット・様子見）"

    return {
        "score": score,
        "comment": comment,
        "rsi": float(rsi),
        "chg5": float(chg5),
        "vola20": float(vola20),
    }