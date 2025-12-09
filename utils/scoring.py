from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================
def _last(series: pd.Series, default: float = np.nan) -> float:
    if series is None or len(series) == 0:
        return float(default)
    v = series.iloc[-1]
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_pct_change(close: pd.Series) -> pd.Series:
    return close.pct_change(fill_method=None)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinanceの hist を受け取り、
    RR計算に必要なベーススコア用指標を付与
    """
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float)

    # 移動平均
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()

    # ボラ20
    ret = _safe_pct_change(close)
    df["vola20"] = ret.rolling(20).std()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 60日高値からの位置
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100.0
    else:
        df["off_high_pct"] = np.nan

    # 流動性
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    return df


# ============================================================
# Sub-score functions (0〜100 scale internally)
# ============================================================
def _trend_quality(df: pd.DataFrame) -> float:
    """
    「波乗りできる状況」にあるか？
    ・MA20 > MA60 (上昇レジーム)
    ・現値の位置
    """
    c = _last(df["Close"])
    ma20 = _last(df["ma20"])
    ma60 = _last(df["ma60"])

    if not np.isfinite(c) or not np.isfinite(ma20) or not np.isfinite(ma60):
        return 30.0  # neutral

    score = 0.0

    # 上昇レジーム
    if ma20 > ma60:
        score += 45.0
    else:
        score += 15.0

    # 現値がMA20から遠すぎない
    d = (c - ma20) / ma20 if ma20 > 0 else 0
    if -0.06 <= d <= 0.04:
        score += 45.0
    elif -0.10 <= d < -0.06 or 0.04 < d <= 0.10:
        score += 25.0
    else:
        score += 10.0

    return float(np.clip(score, 0, 100))


def _pullback_quality(df: pd.DataFrame) -> float:
    """
    押し目の質
    ・RSI
    ・高値からの下落率
    """
    rsi = _last(df["rsi14"])
    off = _last(df["off_high_pct"])

    score = 0.0

    # RSI
    if np.isfinite(rsi):
        if 33 <= rsi <= 50:
            score += 55.0
        elif 27 <= rsi < 33 or 50 < rsi <= 58:
            score += 35.0
        else:
            score += 10.0

    # 高値からの押し
    if np.isfinite(off):
        if -16 <= off <= -5:
            score += 45.0
        elif -25 <= off < -16 or -5 < off <= 5:
            score += 28.0
        else:
            score += 5.0

    return float(np.clip(score, 0, 100))


def _liquidity(df: pd.DataFrame) -> float:
    """
    値幅取りできるか？（薄い銘柄は除外）
    """
    t = _last(df["turnover_avg20"])

    if not np.isfinite(t) or t <= 0:
        return 10.0

    if t >= 3e9:
        return 90.0
    if t >= 1e9:
        return 75.0
    if t >= 2e8:
        return 55.0
    if t >= 1e8:
        return 30.0
    return 10.0


# ============================================================
# Final score (0〜100) → RRエンジンの燃料
# ============================================================
def score_stock(hist: pd.DataFrame) -> float | None:
    """
    RR時代の新スコアリング
    ・Quality / Setup / Liquidity を合成
    ・昔の「A=80点」という概念ではなく、
      「未来の波に乗る力」を0〜100で測る
    """
    if hist is None or len(hist) < 60:
        return None

    df = _add_indicators(hist)

    s_trend = _trend_quality(df)       # 0〜100
    s_pull = _pullback_quality(df)     # 0〜100
    s_liq = _liquidity(df)             # 0〜100

    # 重み
    w_trend = 0.45
    w_pull = 0.40
    w_liq = 0.15

    raw = s_trend * w_trend + s_pull * w_pull + s_liq * w_liq

    # scale to 0〜100
    score = float(np.clip(raw, 0, 100))

    return score