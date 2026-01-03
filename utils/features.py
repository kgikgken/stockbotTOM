from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.util import safe_float


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return float("nan")
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return safe_float(v)


def _rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return safe_float(rsi.iloc[-1])


def _ma(close: pd.Series, n: int) -> float:
    if close is None or len(close) < n:
        return safe_float(close.iloc[-1])
    return safe_float(close.rolling(n).mean().iloc[-1])


def _slope_pct(series: pd.Series, n: int = 5) -> float:
    if series is None or len(series) < n + 1:
        return float("nan")
    a = safe_float(series.iloc[-1 - n])
    b = safe_float(series.iloc[-1])
    if not np.isfinite(a) or a == 0:
        return float("nan")
    return (b / a - 1.0)


def macro_tag_from_sector(sector: str) -> str:
    s = (sector or "").strip()
    # 日本語セクターのざっくり分類
    rate = ("銀行", "保険", "その他金融", "不動産")
    cyc = ("非鉄", "鉱業", "機械", "建設", "輸送", "金属", "化学", "電気機器")
    defn = ("食料", "医薬", "陸運", "電力", "ガス", "水産", "農林")
    growth = ("情報", "通信", "サービス")

    if any(k in s for k in rate):
        return "rate_sensitive"
    if any(k in s for k in defn):
        return "defensive"
    if any(k in s for k in growth):
        return "growth"
    if any(k in s for k in cyc):
        return "cyclical"
    return "other"


@dataclass
class FeaturePack:
    close: float
    open_: float
    prev_close: float
    ma20: float
    ma50: float
    ma10: float
    rsi14: float
    atr: float
    atr_pct: float
    adv20_jpy: float
    vol_last: float
    vol_ma20: float
    trend_ok: bool
    pullback_dist_atr: float
    gap_atr: float


def compute_features(hist: pd.DataFrame) -> FeaturePack:
    df = hist.copy()
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)

    c = safe_float(close.iloc[-1])
    o = safe_float(open_.iloc[-1])
    pc = safe_float(close.iloc[-2]) if len(close) >= 2 else c

    ma20 = _ma(close, 20)
    ma50 = _ma(close, 50)
    ma10 = _ma(close, 10)

    rsi = _rsi(close, 14)
    atr = _atr(df, 14)
    atr_pct = (atr / c * 100.0) if np.isfinite(atr) and np.isfinite(c) and c > 0 else float("nan")

    # 売買代金（JPY代理）: Close*Volume
    turnover = close * vol
    adv20 = safe_float(turnover.rolling(20).mean().iloc[-1]) if len(turnover) >= 20 else safe_float(turnover.mean())
    vol_last = safe_float(vol.iloc[-1])
    vol_ma20 = safe_float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else safe_float(vol.mean())

    trend_ok = bool(np.isfinite(c) and np.isfinite(ma20) and np.isfinite(ma50) and c > ma20 > ma50)

    # pullback distance vs MA20 in ATR units
    pullback_dist_atr = float("nan")
    if np.isfinite(atr) and atr > 0 and np.isfinite(ma20) and np.isfinite(c):
        pullback_dist_atr = abs(c - ma20) / atr

    # gap risk in ATR units (using last day's open vs prev close as proxy)
    gap_atr = float("nan")
    if np.isfinite(atr) and atr > 0 and np.isfinite(o) and np.isfinite(pc):
        gap_atr = (o - pc) / atr

    return FeaturePack(
        close=c,
        open_=o,
        prev_close=pc,
        ma20=ma20,
        ma50=ma50,
        ma10=ma10,
        rsi14=rsi,
        atr=atr,
        atr_pct=atr_pct,
        adv20_jpy=adv20,
        vol_last=vol_last,
        vol_ma20=vol_ma20,
        trend_ok=trend_ok,
        pullback_dist_atr=pullback_dist_atr,
        gap_atr=gap_atr,
    )


def normalize01(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return float(np.clip(v, 0.0, 1.0))


def estimate_pwin(
    fp: FeaturePack,
    sector_rank: int | None,
    liquidity_floor: float = 200_000_000.0,
) -> float:
    """
    代理特徴で Pwin を推定（0-1）。
    目的：勝率ではなく EV の “現実値” を作る。
    """
    # trend strength: MA構造＋MA20傾き代理（close/ma20）
    trend = normalize01(fp.close / fp.ma20 if np.isfinite(fp.ma20) and fp.ma20 > 0 else float("nan"), 0.98, 1.05)

    # pullback quality: MA20距離が小さいほど良い（0〜0.8ATR）
    pb = 1.0 - normalize01(fp.pullback_dist_atr, 0.0, 0.8)

    # RSI: 40-62が中心
    rsi_score = 1.0 - abs(normalize01(fp.rsi14, 30, 80) - normalize01(52, 30, 80))

    # sector rank: 1が最良
    sec = 0.5
    if sector_rank is not None and sector_rank > 0:
        sec = 1.0 - normalize01(sector_rank, 1, 33)

    # volume quality: 押し目は出来高が落ちてる方が良い（vol_last < vol_ma20）
    vq = 1.0 if (np.isfinite(fp.vol_last) and np.isfinite(fp.vol_ma20) and fp.vol_last <= fp.vol_ma20) else 0.4

    # gap risk: GU proxyが大きいと減点
    gap = 1.0 - normalize01(fp.gap_atr, 0.5, 1.2)

    # liquidity: ADV20が高いほど良い
    liq = normalize01(fp.adv20_jpy, liquidity_floor, liquidity_floor * 5.0)

    raw = (
        0.22 * trend
        + 0.18 * pb
        + 0.12 * rsi_score
        + 0.14 * sec
        + 0.14 * vq
        + 0.10 * gap
        + 0.10 * liq
    )
    return float(np.clip(raw, 0.05, 0.75))