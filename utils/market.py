from __future__ import annotations

from datetime import date
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf


INDEX_N225 = "^N225"
INDEX_TOPIX = "^TOPX"


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _fetch(symbol: str, period: str = "240d") -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _ma(series: pd.Series, w: int) -> float:
    if series is None or len(series) < w:
        return _safe_float(series.iloc[-1]) if series is not None and len(series) else np.nan
    return _safe_float(series.rolling(w).mean().iloc[-1])


def _slope20(series: pd.Series) -> float:
    """
    SMA20の傾き（5営業日での%変化）
    """
    if series is None or len(series) < 26:
        return 0.0
    ma20 = series.rolling(20).mean()
    a = _safe_float(ma20.iloc[-6], np.nan)
    b = _safe_float(ma20.iloc[-1], np.nan)
    if not (np.isfinite(a) and np.isfinite(b) and a > 0):
        return 0.0
    return float((b / a - 1.0) * 100.0)


def _gap_freq(df: pd.DataFrame, lookback: int = 30) -> float:
    """
    ギャップ頻度（|Open/PrevClose-1|>1% の比率）
    """
    if df is None or df.empty or len(df) < 10:
        return 0.0
    d = df.tail(lookback).copy()
    if "Open" not in d.columns or "Close" not in d.columns:
        return 0.0
    prev = d["Close"].shift(1)
    gap = (d["Open"] / (prev.replace(0, np.nan)) - 1.0).abs()
    return float(np.nanmean((gap > 0.01).astype(float)))


def _vola20(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 25:
        return 0.0
    c = df["Close"].astype(float)
    r = c.pct_change(fill_method=None)
    return float(np.nan_to_num(r.rolling(20).std().iloc[-1], nan=0.0))


def _score_index(df: pd.DataFrame) -> Dict[str, float]:
    """
    MA構造（20/50）, 短期モメンタム, ボラ, ギャップ頻度で 0-100 を作る
    """
    if df is None or df.empty or len(df) < 80:
        return {"score": 50.0, "mom5": 0.0, "mom20": 0.0, "trend": 0.0, "gapf": 0.0, "vola": 0.0}

    c = df["Close"].astype(float)
    last = float(c.iloc[-1])
    ma20 = _ma(c, 20)
    ma50 = _ma(c, 50)
    slope = _slope20(c)
    mom5 = float((c.iloc[-1] / c.iloc[-6] - 1.0) * 100.0) if len(c) >= 6 else 0.0
    mom20 = float((c.iloc[-1] / c.iloc[-21] - 1.0) * 100.0) if len(c) >= 21 else 0.0
    gapf = _gap_freq(df, 30)
    vola = _vola20(df)

    trend = 0.0
    if np.isfinite(last) and np.isfinite(ma20) and np.isfinite(ma50):
        if last > ma20 > ma50:
            trend = 1.0
        elif last > ma20:
            trend = 0.6
        elif ma20 > ma50:
            trend = 0.4
        else:
            trend = 0.0

    sc = 50.0
    sc += 22.0 * trend
    sc += float(np.clip(mom5, -5, 5)) * 3.0
    sc += float(np.clip(mom20, -10, 10)) * 1.2
    sc += float(np.clip(slope, -2.0, 2.0)) * 7.0

    # 不安定さを減点
    sc -= float(np.clip((gapf - 0.12) * 100.0, 0, 12)) * 1.4
    sc -= float(np.clip((vola - 0.015) * 1000.0, 0, 12)) * 1.0

    return {
        "score": float(np.clip(sc, 0, 100)),
        "mom5": mom5,
        "mom20": mom20,
        "trend": trend,
        "gapf": gapf,
        "vola": vola,
    }


def calc_market_context(today_date: date) -> Dict[str, object]:
    """
    MarketScore 0-100
    NO-TRADE条件（仕様）
      - MarketScore < 45
      - ΔMarketScore_3d ≤ -5 かつ MarketScore < 55
    """
    d_nk = _fetch(INDEX_N225, "240d")
    d_tp = _fetch(INDEX_TOPIX, "240d")

    s1 = _score_index(d_nk)
    s2 = _score_index(d_tp)

    score = float((s1["score"] + s2["score"]) / 2.0)

    def _score_series(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        c = df["Close"].astype(float)
        # rolling score approximations (for 3d delta)
        ma20 = c.rolling(20).mean()
        ma50 = c.rolling(50).mean()
        trend = ((c > ma20) & (ma20 > ma50)).astype(float)
        mom5 = (c / c.shift(5) - 1.0) * 100.0
        mom20 = (c / c.shift(20) - 1.0) * 100.0
        slope = (ma20 / ma20.shift(5) - 1.0) * 100.0
        gapf = ((df["Open"] / df["Close"].shift(1) - 1.0).abs() > 0.01).astype(float).rolling(30).mean()
        vola = c.pct_change(fill_method=None).rolling(20).std()
        sc = 50.0 + 22.0 * trend + np.clip(mom5, -5, 5) * 3.0 + np.clip(mom20, -10, 10) * 1.2 + np.clip(slope, -2, 2) * 7.0
        sc = sc - np.clip((gapf - 0.12) * 100.0, 0, 12) * 1.4 - np.clip((vola - 0.015) * 1000.0, 0, 12) * 1.0
        return sc.clip(0, 100)

    ss1 = _score_series(d_nk)
    ss2 = _score_series(d_tp)
    if len(ss1) >= 4 and len(ss2) >= 4:
        s_today = float((ss1.iloc[-1] + ss2.iloc[-1]) / 2.0)
        s_3d = float((ss1.iloc[-4] + ss2.iloc[-4]) / 2.0)
        chg3 = s_today - s_3d
    else:
        chg3 = 0.0

    score_i = int(np.clip(round(score), 0, 100))

    if score_i >= 70:
        comment = "強い（順張りOK）"
    elif score_i >= 60:
        comment = "やや強い（押し目のみ）"
    elif score_i >= 55:
        comment = "中立（厳選）"
    elif score_i >= 45:
        comment = "弱め（新規かなり絞る）"
    else:
        comment = "弱い（NO-TRADE）"

    no_trade_core = False
    reasons = []
    if score_i < 45:
        no_trade_core = True
        reasons.append("MarketScore<45")
    if chg3 <= -5.0 and score_i < 55:
        no_trade_core = True
        reasons.append("ΔMarketScore_3d<=-5 & Score<55")

    return {
        "score": score_i,
        "delta3d": float(chg3),
        "comment": comment,
        "no_trade_core": bool(no_trade_core),
        "no_trade_reasons": reasons,
        "details": {"n225": s1, "topix": s2},
    }
