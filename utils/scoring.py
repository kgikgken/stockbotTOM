import numpy as np
import pandas as pd

# ============================================================
# Coreスコア計算（Aランク / Bランク）
# ============================================================

def calc_core_score(series: pd.Series) -> dict:
    """
    series: yfinance.Ticker(t).history(period="3mo") の 1銘柄のSeries
    戻り値:
      {
        "score": 数値,
        "rank": "A" or "B" or None,
        "reason": "簡易コメント"
      }
    """

    close = series["Close"].iloc[-1]
    high_20 = series["High"].rolling(20).max().iloc[-1]
    low_20 = series["Low"].rolling(20).min().iloc[-1]

    # ====== トレンド強度（20MAの傾き） ======
    ma20 = series["Close"].rolling(20).mean()
    trend = ma20.iloc[-1] - ma20.iloc[-5]

    trend_score = np.clip((trend / close) * 400, -20, 30)

    # ====== 押し目の質（RSI + 高値からの下落率） ======
    delta = series["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rsi = 100 * up / (up + down + 1e-9)
    rsi_now = rsi.iloc[-1]

    if rsi_now < 35:
        rsi_score = 25
    elif rsi_now < 45:
        rsi_score = 15
    else:
        rsi_score = -10

    drop_rate = (close - high_20) / high_20
    drop_score = np.clip(-drop_rate * 80, -10, 25)

    # ====== ボラティリティ（安定度） ======
    vol = np.std(series["Close"].pct_change().dropna())
    vol_score = np.clip((0.02 - vol) * 300, -20, 20)

    # ====== 総合スコア ======
    score = trend_score + rsi_score + drop_score + vol_score
    score = float(np.clip(score, -20, 100))

    # ====== ランク判定 ======
    if score >= 75:
        rank = "A"
    elif score >= 60:
        rank = "B"
    else:
        rank = None

    # ====== 理由 ======
    reason = []
    if trend_score > 10:
        reason.append("トレンド◯")
    if rsi_now < 45:
        reason.append(f"RSI押し目({int(rsi_now)})")
    if vol_score > 10:
        reason.append("安定度◯")

    return {
        "score": score,
        "rank": rank,
        "reason": " / ".join(reason) if reason else ""
    }