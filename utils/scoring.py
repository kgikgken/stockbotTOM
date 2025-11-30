import numpy as np
import pandas as pd


# スコア計算（100点満点）
def score_stock(hist: pd.DataFrame) -> int:
    close = hist["Close"]

    # ① トレンド（20MAの傾き）
    ma20 = close.rolling(20).mean()
    trend = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5] * 100 if ma20.iloc[-5] != 0 else 0
    trend_score = np.clip(trend * 2, -20, 20)

    # ② 押し目の質
    rsi = calc_rsi(close)
    rsi_score = 20 - abs(rsi - 40)

    # ③ 高値からの下落率
    peak = close.max()
    drop = (peak - close.iloc[-1]) / peak * 100
    drop_score = np.clip(drop, 0, 20)

    # ④ 流動性（ボラ＋売買代金）
    vol = hist["Volume"].iloc[-20:].mean()
    vol_score = np.log10(max(vol, 1)) * 5
    vol_score = np.clip(vol_score, 0, 20)

    total = trend_score + rsi_score + drop_score + vol_score
    return int(np.clip(total, 0, 100))


def classify_core(score: int) -> str:
    if score >= 75:
        return "A"
    elif score >= 60:
        return "B"
    else:
        return "N"  # 候補外


def calc_rsi(close: pd.Series, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    return rsi.iloc[-1]
