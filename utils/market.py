from __future__ import annotations

import yfinance as yf
import pandas as pd
import numpy as np


INDEX_TICKER = "^N225"   # 日経平均（TOPIXに替えてもOK）


def evaluate_market() -> dict:
    df = yf.download(INDEX_TICKER, period="6mo", auto_adjust=True)
    close = df["Close"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    score = 50

    # トレンド構造
    if close.iloc[-1] > sma50.iloc[-1]:
        score += 10
    if sma20.iloc[-1] > sma50.iloc[-1]:
        score += 10

    # モメンタム
    ret5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100
    ret20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100
    score += np.clip(ret5, -5, 5)
    score += np.clip(ret20 / 2, -5, 5)

    # 変化率
    delta_3d = score - (
        50
        + np.clip(
            ((close.iloc[-4] / close.iloc[-9]) - 1) * 100, -5, 5
        )
    )

    # NO-TRADE判定
    no_trade = False
    reason = ""

    if score < 45:
        no_trade = True
        reason = "MarketScore<45"
    elif delta_3d <= -5 and score < 55:
        no_trade = True
        reason = "Δ3d<=-5 & MarketScore<55"

    # レバ調整
    if score >= 65:
        leverage = "2.0倍（強気）"
    elif score >= 55:
        leverage = "1.5倍（中立）"
    else:
        leverage = "1.1倍（守り）"

    return {
        "score": int(score),
        "delta_3d": int(delta_3d),
        "no_trade": no_trade,
        "no_trade_reason": reason,
        "leverage": leverage,
        "comment": "強め" if score >= 60 else "中立" if score >= 50 else "弱め",
        "regime_multiplier": 1.05 if score >= 60 else 0.8 if score < 50 else 1.0,
    }