from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# ============================================================
# ヘルパー
# ============================================================
def _last(series: pd.Series, default: float = np.nan) -> float:
    if series is None or len(series) == 0:
        return float(default)
    v = series.iloc[-1]
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_pct_change(close: pd.Series, window: int = 20) -> pd.Series:
    # yfinance の仕様変更対策で fill_method=None を明示
    return close.pct_change(fill_method=None)


# ============================================================
# インジケータ計算
# ============================================================
def _add_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float)

    # MA
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ボラ20
    ret = _safe_pct_change(close, 20)
    df["vola20"] = ret.rolling(20).std()

    # ATR14
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # 60日高値からの位置
    if len(close) >= 60:
        rolling_high = close.rolling(60).max()
        df["off_high_pct"] = (close - rolling_high) / rolling_high * 100.0
    else:
        df["off_high_pct"] = np.nan

    # 下ヒゲ比率
    rng = high - low
    lower_shadow = np.where(close >= open_, close - low, open_ - low)
    df["lower_shadow_ratio"] = np.where(rng > 0, lower_shadow / rng, 0.0)

    # 流動性
    df["turnover"] = close * vol
    df["turnover_avg20"] = df["turnover"].rolling(20).mean()

    return df


# ============================================================
# 各コンポーネント
# ============================================================
def _compute_entry_price(df: pd.DataFrame) -> float:
    """
    3〜10日スイング用のIN価格（安全寄り）
    - ベースはMA20
    - 強トレンドでは少し上寄せ
    - 直近安値を極端に割らない
    """
    close = df["Close"].astype(float)
    price = _last(close)
    ma5 = _last(df["ma5"], default=price)
    ma20 = _last(df["ma20"], default=price)
    atr = _last(df["atr14"], default=0.0)

    if not np.isfinite(price) or price <= 0:
        return 0.0

    # 基本は MA20 付近
    entry = ma20 if np.isfinite(ma20) and ma20 > 0 else price

    # ATR で少し下にずらす（押し目を待つ）
    if np.isfinite(atr) and atr > 0:
        entry = entry - atr * 0.4

    # 強い上昇トレンドなら少し上寄せ（深追いしすぎない）
    if price > ma5 > ma20:
        entry = ma20 + (ma5 - ma20) * 0.35

    # 現値より上になってしまったら、現値少し下に補正
    if entry > price:
        entry = price * 0.995

    # 直近安値より極端に下には置かない
    tail = close.tail(10)
    if len(tail) > 0:
        last_low = float(tail.min())
        if entry < last_low * 0.97:
            entry = last_low * 0.97

    return float(round(entry, 1))


def _base_tp_sl_from_vola(vola20: float) -> (float, float):
    """
    ボラからベースのTP/SLを決める（ハイブリッド用）
    戻り値: (tp_pct, sl_pct<0)
    """
    v = abs(vola20) if np.isfinite(vola20) else 0.025

    if v < 0.015:
        tp = 0.06
        sl = -0.03
    elif v < 0.03:
        tp = 0.08
        sl = -0.04
    elif v < 0.05:
        tp = 0.10
        sl = -0.05
    else:
        tp = 0.12
        sl = -0.06

    return float(tp), float(sl)


def _adjust_by_market(tp: float, sl: float, mkt_score: int) -> (float, float):
    """
    地合いでTP/SLを調整
    """
    t, s = tp, sl

    if mkt_score >= 70:
        t *= 1.15
        s *= 0.9
    elif mkt_score >= 60:
        t *= 1.05
    elif mkt_score >= 50:
        # 中立
        pass
    elif mkt_score >= 40:
        t *= 0.9
        s *= 0.9
    else:
        t *= 0.8
        s *= 0.85

    return float(t), float(s)


def _pullback_factor(df: pd.DataFrame) -> float:
    """
    押し目完成度 0.7〜1.4（C = ハイブリッド）
    """
    rsi = _last(df["rsi14"])
    off = _last(df["off_high_pct"])
    ma20 = _last(df["ma20"])
    price = _last(df["Close"])
    shadow = _last(df["lower_shadow_ratio"])

    if not np.isfinite(price) or price <= 0:
        return 1.0

    score = 0.0

    # RSI
    if np.isfinite(rsi):
        if 35 <= rsi <= 50:
            score += 35.0
        elif 30 <= rsi < 35 or 50 < rsi <= 55:
            score += 22.0
        elif 25 <= rsi < 30 or 55 < rsi <= 60:
            score += 10.0
        else:
            score += 4.0

    # 高値からの押し
    if np.isfinite(off):
        if -18 <= off <= -5:
            score += 35.0
        elif -25 <= off < -18 or -5 < off <= 5:
            score += 20.0
        else:
            score += 8.0

    # MA20 との距離
    if np.isfinite(ma20) and ma20 > 0:
        d = (price - ma20) / ma20
        if -0.03 <= d <= 0.02:
            score += 20.0
        elif -0.06 <= d < -0.03 or 0.02 < d <= 0.05:
            score += 10.0
        else:
            score += 3.0

    # 下ヒゲ
    if np.isfinite(shadow):
        if shadow >= 0.5:
            score += 10.0
        elif shadow >= 0.3:
            score += 5.0
        else:
            score += 1.0

    # 0〜100 → 0.7〜1.4
    score = float(np.clip(score, 0.0, 100.0))
    factor = 0.7 + 0.7 * (score / 100.0)
    return float(np.clip(factor, 0.7, 1.4))


def _liquidity_factor(df: pd.DataFrame) -> float:
    """
    板厚・流動性による補正 0.8〜1.1
    """
    t = _last(df["turnover_avg20"])

    if not np.isfinite(t) or t <= 0:
        return 0.9

    if t >= 3e9:
        f = 1.10
    elif t >= 1e9:
        f = 1.05
    elif t >= 2e8:
        f = 1.00
    elif t >= 1e8:
        f = 0.90
    else:
        f = 0.80

    return float(f)


def _risk_adjust_sl(base_sl: float, pull_factor: float) -> float:
    """
    押し目の質で損切りを上下する（浅く / 深く）
    """
    s = base_sl

    if pull_factor >= 1.15:
        s *= 0.85  # いい押し目 → 損切浅く
    elif pull_factor <= 0.85:
        s *= 1.20  # 中途半端な押し目 → 損切り深め（外されにくく）
    else:
        # そのまま
        pass

    return float(s)


# ============================================================
# メイン：RR計算
# ============================================================
def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int) -> Dict[str, float]:
    """
    RRハイブリッド版
    戻り値:
      {
        "entry": entry_price,
        "tp_pct": tp_pct,   # +0.10 → +10%
        "sl_pct": sl_pct,   # -0.04 → -4%
        "rr": rr_value      # TP/|SL|
      }
    """
    if hist is None or len(hist) < 40:
        return {"entry": 0.0, "tp_pct": 0.08, "sl_pct": -0.04, "rr": 2.0}

    df = _add_indicators(hist)

    close = df["Close"].astype(float)
    price = _last(close)
    vola20 = _last(df["vola20"], default=0.025)

    # Entry
    entry = _compute_entry_price(df)
    if entry <= 0 and np.isfinite(price) and price > 0:
        entry = price

    # Base TP/SL
    tp_base, sl_base = _base_tp_sl_from_vola(vola20)
    tp_mkt, sl_mkt = _adjust_by_market(tp_base, sl_base, int(mkt_score))

    # Factors
    f_pull = _pullback_factor(df)
    f_liq = _liquidity_factor(df)

    # Apply factors
    tp = tp_mkt * f_pull * f_liq
    sl = _risk_adjust_sl(sl_mkt, f_pull)

    # 安全バンド
    tp = float(np.clip(tp, 0.05, 0.20))
    sl = float(np.clip(sl, -0.08, -0.02))

    # RR
    if not (np.isfinite(tp) and np.isfinite(sl) and sl < 0):
        rr = 1.5
    else:
        rr = float(tp / abs(sl))
        rr = float(np.clip(rr, 0.8, 4.0))

    return {
        "entry": float(entry),
        "tp_pct": tp,
        "sl_pct": sl,
        "rr": rr,
    }