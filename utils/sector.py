from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================
# 内部ヘルパー
# ============================================================

def _last_val(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return np.nan
    v = series.iloc[-1]
    try:
        return float(v)
    except Exception:
        return np.nan


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance の history を受け取り、
    スコア&IN判定に使う指標を全部載せる。
    """
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float)

    # 移動平均
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 20日ボラ（リターンの標準偏差）
    ret = close.pct_change(fill_method=None)
    df["vola20"] = ret.rolling(20).std()

    # 60日高値からの位置
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100
    else:
        df["off_high_pct"] = np.nan

    # 出来高・売買代金
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    return df


# ============================================================
# Core Score 0〜100
# ============================================================

def _trend_score(df: pd.DataFrame) -> float:
    close = df["Close"].astype(float)
    ma20 = df["ma20"]
    ma50 = df["ma50"]

    sc = 0.0
    c_last = _last_val(close)
    ma20_last = _last_val(ma20)
    ma50_last = _last_val(ma50)

    # MAの傾き
    slope = _last_val(ma20.pct_change(fill_method=None))
    if np.isfinite(slope):
        if slope >= 0.01:
            sc += 8
        elif slope > 0:
            sc += 4 + (slope / 0.01) * 4
        else:
            sc += max(0.0, 4 + slope * 50)  # -0.08 でほぼ0

    # MAの並び
    if np.isfinite(c_last) and np.isfinite(ma20_last) and np.isfinite(ma50_last):
        if c_last > ma20_last > ma50_last:
            sc += 8
        elif c_last > ma20_last:
            sc += 4
        elif ma20_last > ma50_last:
            sc += 2

    # 高値からの位置（押し目過ぎも少し減点）
    off = _last_val(df["off_high_pct"])
    if np.isfinite(off):
        if -5 <= off <= 0:
            sc += 4
        elif -15 <= off < -5:
            sc += 3
        elif -25 <= off < -15:
            sc += 1

    return float(np.clip(sc, 0, 20))


def _pullback_score(df: pd.DataFrame) -> float:
    rsi = _last_val(df["rsi14"])
    off = _last_val(df["off_high_pct"])

    sc = 0.0

    # RSI
    if np.isfinite(rsi):
        if 30 <= rsi <= 45:
            sc += 8
        elif 45 < rsi <= 60:
            sc += 4
        elif 20 <= rsi < 30:
            sc += 3

    # 高値からの下落率
    if np.isfinite(off):
        if -18 <= off <= -5:
            sc += 8
        elif -25 <= off < -18:
            sc += 4

    return float(np.clip(sc, 0, 20))


def _liquidity_score(df: pd.DataFrame) -> float:
    t = _last_val(df["turnover_avg20"])
    v = _last_val(df["vola20"])
    sc = 0.0

    # 流動性
    if np.isfinite(t):
        if t >= 10e8:
            sc += 16
        elif t >= 1e8:
            sc += 16 * (t - 1e8) / 9e8

    # ボラ（極端でなければ加点）
    if np.isfinite(v):
        if 0.015 <= v <= 0.06:
            sc += 4

    return float(np.clip(sc, 0, 20))


def score_stock(hist: pd.DataFrame) -> float | None:
    """
    Coreスコア（0〜100）
    """
    if hist is None or len(hist) < 60:
        return None

    df = _add_indicators(hist)

    ts = _trend_score(df)
    ps = _pullback_score(df)
    ls = _liquidity_score(df)

    raw = ts + ps + ls  # 最大60
    if not np.isfinite(raw):
        return None

    score = float(raw / 60.0 * 100.0)
    return float(np.clip(score, 0, 100))


# ============================================================
# INランク ＆ TP/SL（％）
# ============================================================

def calc_vola20(hist: pd.DataFrame) -> float:
    if hist is None or len(hist) < 21:
        return np.nan
    close = hist["Close"].astype(float)
    ret = close.pct_change(fill_method=None)
    vola20 = ret.rolling(20).std().iloc[-1]
    try:
        return float(vola20)
    except Exception:
        return np.nan


def _classify_vola(vola: float) -> str:
    if not np.isfinite(vola):
        return "mid"
    if vola < 0.02:
        return "low"
    if vola > 0.06:
        return "high"
    return "mid"


def calc_inout_for_stock(hist: pd.DataFrame):
    """
    INランク + 利確/損切り目安（％）

    戻り値:
      in_rank: "強IN" / "通常IN" / "弱めIN" / "様子見"
      tp_pct: 利確目安（+◯％）
      sl_pct: 損切り目安（-◯％）
    """
    if hist is None or len(hist) < 40:
        return "様子見", 6.0, -3.0

    df = _add_indicators(hist)

    close_last = _last_val(df["Close"])
    ma20_last = _last_val(df["ma20"])
    rsi_last = _last_val(df["rsi14"])
    off_last = _last_val(df["off_high_pct"])
    vola = calc_vola20(hist)
    vola_class = _classify_vola(vola)

    # --- ベースINランク ---
    rank = "様子見"

    if (
        np.isfinite(rsi_last)
        and np.isfinite(ma20_last)
        and np.isfinite(close_last)
        and np.isfinite(off_last)
    ):
        # 理想押し目
        if 30 <= rsi_last <= 45 and -18 <= off_last <= -5 and close_last >= ma20_last * 0.97:
            rank = "強IN"
        # 普通の押し目 or トレンド押し
        elif 40 <= rsi_last <= 60 and -15 <= off_last <= 5 and close_last >= ma20_last * 0.99:
            rank = "通常IN"
        # 少し無理する押し目
        elif 25 <= rsi_last < 30 or 60 < rsi_last <= 70:
            rank = "弱めIN"
        else:
            rank = "様子見"
    else:
        rank = "様子見"

    # --- TP/SLベース（％表記） ---
    tp = 8.0   # +8%
    sl = -4.0  # -4%

    if vola_class == "low":
        tp = 6.0
        sl = -3.0
    elif vola_class == "high":
        tp = 12.0
        sl = -6.0

    # ランクに応じた微調整
    if rank == "強IN":
        tp *= 1.2
        sl *= 0.9
    elif rank == "弱めIN":
        tp *= 0.9
        sl *= 0.9
    elif rank == "様子見":
        tp *= 0.8
        sl *= 0.8

    tp = float(np.clip(tp, 4.0, 20.0))
    sl = float(np.clip(sl, -10.0, -2.0))

    return rank, tp, sl