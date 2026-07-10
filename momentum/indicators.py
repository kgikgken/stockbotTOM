"""モメンタム系テクニカル指標 — 全て日次OHLCVのみから計算(yfinance単一ソース前提).

歪み系(mispricing/indicators.py)とは独立。ATR/SMAの純粋な数学的ユーティリティのみ
共有元から再利用し、モメンタム固有の指標(ADX・12-1モメンタム・52週高値近接度・
相対強度・VCP収縮・ドンチアンブレイク・シャンデリア水準)はここに実装する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mispricing.indicators import sma, atr_wilder  # 純粋な数学ユーティリティのみ再利用


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)

    atr_n = tr.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean() / atr_n.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean() / atr_n.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


def chandelier_exit_long(high: pd.Series, atr_series: pd.Series, n: int, mult: float) -> pd.Series:
    """直近n日高値 − mult×ATR(n)。ロング用の可変トレーリングストップ水準。"""
    return high.rolling(n, min_periods=n).max() - mult * atr_series


def donchian_high(high: pd.Series, n: int) -> pd.Series:
    return high.rolling(n, min_periods=n).max()


def vcp_contraction(high: pd.Series, low: pd.Series, lookback: int, ratio: float) -> pd.Series:
    """直近半分の値幅 vs 直近全体の値幅(いずれも前日までで評価し、当日のブレイク足自体を含めない)。"""
    rng = (high - low)
    recent = rng.rolling(lookback // 2, min_periods=lookback // 2).mean().shift(1)
    baseline = rng.rolling(lookback, min_periods=lookback).mean().shift(1)
    return recent / baseline.replace(0, np.nan) <= ratio


def compute_momentum_features(df: pd.DataFrame, bench_logclose: pd.Series | None, cfg) -> dict | None:
    """df: OHLCV daily。bench_logclose: ベンチマーク(TOPIX)のlog(close)系列(日付整列前)。"""
    if df is None or len(df) < max(cfg.regime_mom_days, 260) + 5:
        return None
    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 260:
        return None

    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    close_now = float(c.iloc[-1])

    sma10, sma20 = sma(c, cfg.pullback_sma_fast), sma(c, cfg.pullback_sma_slow)
    sma50, sma150, sma200 = sma(c, 50), sma(c, 150), sma(c, 200)
    atr_n = atr_wilder(h, l, c, cfg.atr_period)
    adx_n = adx(h, l, c, cfg.adx_period)
    chand = chandelier_exit_long(h, atr_n, cfg.atr_period, cfg.chandelier_mult)
    donch_prev = donchian_high(h, cfg.donchian_days).shift(1)  # 前日までのN日高値(当日ブレイク判定用)
    vcp_now = bool(vcp_contraction(h, l, cfg.vcp_lookback, cfg.vcp_contraction_ratio).iloc[-1])

    vmean20 = float(v.iloc[-21:-1].mean())
    vsd20 = float(v.iloc[-21:-1].std(ddof=0))
    vol_ratio_today = float(v.iloc[-1]) / vmean20 if vmean20 > 0 else np.nan

    # --- モメンタム総合スコア構成要素 ---
    logc = np.log(c)
    mom_12_1 = float(logc.shift(21).iloc[-1] - logc.shift(252).iloc[-1]) if len(c) > 252 else np.nan
    high52w = float(c.iloc[-252:].max())
    high52w_proximity = close_now / high52w if high52w > 0 else np.nan

    rel_strength = np.nan
    if bench_logclose is not None:
        s_now, s_now_al = logc.align(bench_logclose, join="inner")
        if len(s_now) > 130:
            stock_ret126 = float(s_now.iloc[-1] - s_now.iloc[-127])
            bench_ret126 = float(s_now_al.iloc[-1] - s_now_al.iloc[-127])
            rel_strength = stock_ret126 - bench_ret126

    trend_align = bool(close_now > sma50.iloc[-1] > sma150.iloc[-1] > sma200.iloc[-1]) \
        if not (pd.isna(sma50.iloc[-1]) or pd.isna(sma150.iloc[-1]) or pd.isna(sma200.iloc[-1])) else False
    breakdown = bool(close_now < sma(c, cfg.breakdown_sma).iloc[-1])

    adv20_jpy = float((c * v).iloc[-21:-1].mean())

    return {
        "close": close_now, "atr": float(atr_n.iloc[-1]), "adx": float(adx_n.iloc[-1]),
        "sma10": float(sma10.iloc[-1]), "sma20": float(sma20.iloc[-1]),
        "sma50": float(sma50.iloc[-1]), "sma150": float(sma150.iloc[-1]), "sma200": float(sma200.iloc[-1]),
        "chandelier": float(chand.iloc[-1]), "donchian_prev": float(donch_prev.iloc[-1]) if not pd.isna(donch_prev.iloc[-1]) else np.nan,
        "vcp_now": vcp_now, "vol_ratio_today": vol_ratio_today,
        "mom_12_1": mom_12_1, "high52w_proximity": high52w_proximity, "rel_strength": rel_strength,
        "trend_align": trend_align, "breakdown": breakdown,
        "adv20_jpy": adv20_jpy,
        "last_date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
    }


def momentum_score(feat: dict, cfg) -> float:
    """候補プール選定用の総合スコア(z化はプール内で相対的に行うため、ここでは素点)。"""
    parts = []
    if not (feat["mom_12_1"] is None or np.isnan(feat["mom_12_1"])):
        parts.append(("mom_12_1", feat["mom_12_1"] * 100, cfg.w_mom_12_1))
    if not np.isnan(feat["high52w_proximity"]):
        parts.append(("high52w", feat["high52w_proximity"] * 100, cfg.w_high52w))
    if not np.isnan(feat["rel_strength"]):
        parts.append(("relstrength", feat["rel_strength"] * 100, cfg.w_relstrength))
    parts.append(("trend_align", 10.0 if feat["trend_align"] else 0.0, cfg.w_trend_align))
    if not parts:
        return float("-inf")
    total_w = sum(p[2] for p in parts)
    return sum(v * w for _, v, w in parts) / total_w if total_w > 0 else float("-inf")
