# ============================================
# utils/setup.py
# Setup 判定（A1 / A2）
# ============================================

import numpy as np


# --------------------------------------------
# 共通インジケータ取得
# --------------------------------------------
def _ma(series, n):
    if len(series) < n:
        return np.nan
    return series.rolling(n).mean().iloc[-1]


def _atr(df, n=14):
    if len(df) < n + 1:
        return np.nan

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1)),
        ),
    )

    return tr.rolling(n).mean().iloc[-1]


def _rsi(series, n=14):
    if len(series) < n + 1:
        return np.nan

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]


# --------------------------------------------
# Setup A 判定（A1 / A2）
# --------------------------------------------
def judge_setup_A(df):
    """
    Setup A を A1 / A2 に分離

    A1:
      ・強トレンド
      ・浅い押し目
      ・再加速期待

    A2:
      ・トレンド継続
      ・やや深い押し目
      ・時間はかかるがEV成立
    """

    close = df["Close"]

    ma20 = _ma(close, 20)
    ma50 = _ma(close, 50)
    ma20_prev = _ma(close[:-5], 20) if len(close) > 25 else np.nan

    atr = _atr(df)
    rsi = _rsi(close)

    if any(np.isnan(x) for x in [ma20, ma50, atr, rsi]):
        return None

    price = close.iloc[-1]

    # --- トレンド前提 ---
    trend_ok = price > ma20 > ma50
    slope_ok = ma20 > ma20_prev

    if not (trend_ok and slope_ok):
        return None

    # --- 押し目判定 ---
    dist_from_ma20 = abs(price - ma20)
    dist_atr = dist_from_ma20 / atr if atr > 0 else np.inf

    # RSI 過熱排除
    if rsi >= 65:
        return None

    # --------------------
    # A1: 浅い押し目
    # --------------------
    if dist_atr <= 0.5 and 40 <= rsi <= 60:
        return {
            "setup": "A1",
            "comment": "浅い押し目・再加速型"
        }

    # --------------------
    # A2: やや深い押し目
    # --------------------
    if 0.5 < dist_atr <= 1.0 and 35 <= rsi <= 55:
        return {
            "setup": "A2",
            "comment": "深め押し目・時間許容型"
        }

    return None


# --------------------------------------------
# Setup 判定エントリポイント
# --------------------------------------------
def detect_setup(df):
    """
    現在は Setup A のみ採用
    将来 Setup B 追加可能
    """

    setup_a = judge_setup_A(df)
    if setup_a:
        return setup_a

    return None