from __future__ import annotations

import numpy as np
import pandas as pd


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def calc_ma(close: pd.Series, window: int) -> float:
    if close is None or len(close) < window:
        return _last_val(close)
    return float(close.rolling(window).mean().iloc[-1])


def _add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    # 60日高値からの距離
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / (rolling_high + 1e-9) * 100
    else:
        df["off_high_pct"] = np.nan

    # 20MAの傾き
    df["trend_slope20"] = df["ma20"].pct_change(fill_method=None)

    # 下ヒゲ比率
    rng = (high - low).replace(0, np.nan)
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    df["lower_shadow_ratio"] = np.where(np.isfinite(rng), lower_shadow / rng, 0.0)

    # 売買代金
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    return df


def score_stock(hist: pd.DataFrame) -> float | None:
    """
    0-100
    """
    if hist is None or len(hist) < 80:
        return None

    df = _add_indicators(hist)

    close = df["Close"].astype(float)
    ma20 = df["ma20"]
    ma50 = df["ma50"]
    slope = df["trend_slope20"]

    c_last = _last_val(close)
    ma20_last = _last_val(ma20)
    ma50_last = _last_val(ma50)
    slope_last = _last_val(slope)

    rsi = _last_val(df["rsi14"])
    off = _last_val(df["off_high_pct"])
    shadow = _last_val(df["lower_shadow_ratio"])
    t = _last_val(df["turnover_avg20"])
    vola = _last_val(df["vola20"])

    sc = 0.0

    # トレンド（最大30）
    if np.isfinite(slope_last):
        if slope_last >= 0.01:
            sc += 12
        elif slope_last > 0:
            sc += 6 + (slope_last / 0.01) * 6
        else:
            sc += max(0.0, 6 + slope_last * 60)

    if np.isfinite(c_last) and np.isfinite(ma20_last) and np.isfinite(ma50_last):
        if c_last > ma20_last > ma50_last:
            sc += 12
        elif c_last > ma20_last:
            sc += 6
        elif ma20_last > ma50_last:
            sc += 3

    # 押し目（最大30）
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            sc += 14
        elif 45 < rsi <= 60:
            sc += 10
        elif 25 <= rsi < 30 or 60 < rsi <= 70:
            sc += 6
        else:
            sc += 2

    if np.isfinite(off):
        if -18 <= off <= -5:
            sc += 10
        elif -25 <= off < -18:
            sc += 6
        elif -5 < off <= 5:
            sc += 6
        else:
            sc += 2

    if np.isfinite(shadow):
        if shadow >= 0.5:
            sc += 6
        elif shadow >= 0.3:
            sc += 3

    # 流動性・ボラ（最大40）
    if np.isfinite(t):
        if t >= 1e9:
            sc += 28
        elif t >= 1e8:
            sc += 28 * (t - 1e8) / 9e8

    if np.isfinite(vola):
        if vola < 0.02:
            sc += 12
        elif vola < 0.06:
            sc += 12 * (0.06 - vola) / 0.04

    score = float(np.clip(sc, 0, 100))
    return score


def calc_inout_for_stock(hist: pd.DataFrame):
    """
    in_rank, tp_pct(%), sl_pct(%; negative)
    """
    if hist is None or len(hist) < 80:
        return "様子見", 6.0, -3.0

    df = _add_indicators(hist)

    close_last = _last_val(df["Close"])
    ma20_last = _last_val(df["ma20"])
    rsi_last = _last_val(df["rsi14"])
    off_last = _last_val(df["off_high_pct"])
    vola = _last_val(df["vola20"])
    shadow = _last_val(df["lower_shadow_ratio"])

    rank = "様子見"

    if np.isfinite(rsi_last) and np.isfinite(ma20_last) and np.isfinite(close_last) and np.isfinite(off_last):
        # “押し目完了”寄り（強）
        if 30 <= rsi_last <= 48 and -20 <= off_last <= -5 and close_last >= ma20_last * 0.97 and shadow >= 0.3:
            rank = "強IN"
        # 普通
        elif 40 <= rsi_last <= 62 and -15 <= off_last <= 5 and close_last >= ma20_last * 0.99:
            rank = "通常IN"
        # 弱め
        elif 25 <= rsi_last < 30 or 62 < rsi_last <= 72:
            rank = "弱めIN"
        else:
            rank = "様子見"

    # TP/SL%（ボラで可変）
    v = float(abs(vola)) if np.isfinite(vola) else 0.03
    if v < 0.015:
        tp = 0.06
        sl = -0.03
    elif v < 0.03:
        tp = 0.08
        sl = -0.04
    else:
        tp = 0.12
        sl = -0.06

    # rank微調整
    if rank == "強IN":
        tp *= 1.05
        sl *= 0.90
    elif rank == "弱めIN":
        tp *= 0.95
        sl *= 0.90
    elif rank == "様子見":
        tp *= 0.85
        sl *= 0.85

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.08, -0.02))

    return rank, tp * 100.0, sl * 100.0