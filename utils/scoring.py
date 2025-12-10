from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# helper
# ============================================================
def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float)

    # MA
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ボラ20
    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    # 60日高値からの位置
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100.0
    else:
        df["off_high_pct"] = np.nan

    # 出来高関連
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    return df


# ============================================================
# Coreスコア（形のスコア）
# ============================================================
def score_stock(hist: pd.DataFrame) -> float:
    """
    銘柄のCoreスコア（0〜100）
    Aランク: score >= 60
    """
    if hist is None or len(hist) < 60:
        return 0.0

    df = _add_indicators(hist)

    score = 0.0

    # トレンド（ma20の傾き＋位置）
    ma20 = df["ma20"]
    slope20 = ma20.pct_change(fill_method=None)
    s_last = _last_val(slope20)
    c_last = _last_val(df["Close"])
    ma20_last = _last_val(ma20)
    ma50_last = _last_val(df["ma50"])

    if np.isfinite(s_last):
        if s_last >= 0.01:
            score += 18.0
        elif s_last > 0:
            score += 8.0 + (s_last / 0.01) * 10.0
        else:
            score += max(0.0, 8.0 + s_last * 120.0)

    if np.isfinite(c_last) and np.isfinite(ma20_last) and np.isfinite(ma50_last):
        if c_last > ma20_last > ma50_last:
            score += 10.0
        elif c_last > ma20_last:
            score += 6.0
        elif ma20_last > ma50_last:
            score += 3.0

    # 押し目状態
    off = _last_val(df["off_high_pct"])
    rsi = _last_val(df["rsi14"])

    if np.isfinite(off):
        if -18.0 <= off <= -5.0:
            score += 10.0
        elif -25.0 <= off < -18.0:
            score += 5.0
        elif -5.0 < off <= 5.0:
            score += 4.0

    if np.isfinite(rsi):
        if 30.0 <= rsi <= 50.0:
            score += 12.0
        elif 25.0 <= rsi < 30.0 or 50.0 < rsi <= 60.0:
            score += 6.0

    # 流動性
    t20 = _last_val(df["turnover_avg20"])
    if np.isfinite(t20):
        if t20 >= 1e9:
            score += 20.0
        elif t20 >= 1e8:
            score += 20.0 * (t20 - 1e8) / 9e8

    # ボラ
    vola20 = _last_val(df["vola20"])
    if np.isfinite(vola20):
        if 0.015 <= vola20 <= 0.06:
            score += 10.0
        elif 0.01 <= vola20 < 0.015 or 0.06 < vola20 <= 0.08:
            score += 5.0

    return float(np.clip(score, 0.0, 100.0))


# ============================================================
# INゾーン判定
# ============================================================
def compute_in_rank(hist: pd.DataFrame) -> str:
    """
    強IN / 通常IN / 弱めIN / 様子見
    """
    if hist is None or len(hist) < 40:
        return "様子見"

    df = _add_indicators(hist)

    close_last = _last_val(df["Close"])
    ma20_last = _last_val(df["ma20"])
    rsi_last = _last_val(df["rsi14"])
    off_last = _last_val(df["off_high_pct"])

    if not all(np.isfinite(x) for x in [close_last, ma20_last, rsi_last, off_last]):
        return "様子見"

    # 条件ベース
    # 強IN
    if (
        close_last >= ma20_last * 0.97
        and 30.0 <= rsi_last <= 50.0
        and -18.0 <= off_last <= -5.0
    ):
        return "強IN"

    # 通常IN
    if (
        close_last >= ma20_last * 0.96
        and 30.0 <= rsi_last <= 55.0
        and -22.0 <= off_last <= 5.0
    ):
        return "通常IN"

    # 弱めIN（少し無理）
    if (
        close_last >= ma20_last * 0.95
        and 25.0 <= rsi_last <= 60.0
        and -25.0 <= off_last <= 8.0
    ):
        return "弱めIN"

    return "様子見"