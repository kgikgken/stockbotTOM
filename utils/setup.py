from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils.util import sma, ema, rsi14, atr14, adv20, atr_pct_last, safe_float, clamp

@dataclass
class SetupInfo:
    setup: str
    tier: int
    entry_low: float
    entry_high: float
    sl: float
    tp2: float
    tp1: float
    rr: float
    expected_days: float
    rday: float
    trend_strength: float
    pullback_quality: float
    gu: bool
    breakout_line: Optional[float] = None

def _trend_strength(c: pd.Series, ma20: pd.Series, ma50: pd.Series) -> float:
    c_last = safe_float(c.iloc[-1], np.nan)
    m20 = safe_float(ma20.iloc[-1], np.nan)
    m50 = safe_float(ma50.iloc[-1], np.nan)
    if not (np.isfinite(c_last) and np.isfinite(m20) and np.isfinite(m50)):
        return 0.0

    slope = safe_float(ma20.pct_change(fill_method=None).iloc[-1], 0.0)

    if c_last > m20 > m50:
        pos = 1.0
    elif c_last > m20:
        pos = 0.7
    elif m20 > m50:
        pos = 0.5
    else:
        pos = 0.3

    ang = clamp((slope / 0.004), -1.0, 1.5)
    base = 0.85 + 0.20 * pos + 0.10 * ang
    return float(clamp(base, 0.80, 1.20))

def _pullback_quality(
    df: pd.DataFrame,
    atr: float,
    ema25: pd.Series,
    ema50: pd.Series,
) -> float:
    """
    押し目の健全性を 0.60〜1.40 でスコア化（高いほど良い）。
    目的：高値更新後の「初押し」や「健全な調整」を優先する。
    """
    if atr <= 0 or len(df) < 40:
        return 0.6

    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    v = df["Volume"].fillna(0)

    # 直近20本の高値更新を検出（過去20本高値の上抜けが1回でもあれば採用）
    hi20_prev = h.rolling(20).max().shift(1)
    breakout = h > hi20_prev

    # ブレイクがない場合は弱い押し目として扱う
    if not breakout.iloc[-20:].any():
        base = 0.75
        # EMA付近で支えられていれば少し加点
        e25 = safe_float(ema25.iloc[-1], np.nan)
        e50 = safe_float(ema50.iloc[-1], np.nan)
        cl = safe_float(c.iloc[-1], np.nan)
        if np.isfinite(e25) and np.isfinite(e50) and np.isfinite(cl):
            if cl >= e25 * 0.98 and cl >= e50 * 0.96:
                base += 0.10
        return float(clamp(base, 0.6, 1.4))

    # 最新のブレイク位置（直近から遡って最初に見つかったブレイク）
    bidx = int(np.where(breakout.values)[0][-1])
    bars_since = (len(df) - 1) - bidx

    score = 0.60

    # ブレイク後 3〜7本の調整を最優先（=初押し/健全調整）
    if 3 <= bars_since <= 7:
        score += 0.35
    elif 1 <= bars_since <= 12:
        score += 0.15

    # 調整の深さ：ブレイク高値からの押しが 0.3〜1.2 ATR が理想
    bh = safe_float(h.iloc[bidx], np.nan)
    pl = float(np.nanmin(l.iloc[bidx:])) if bidx < len(df) else np.nan
    if np.isfinite(bh) and np.isfinite(pl):
        depth_atr = (bh - pl) / atr
        if 0.30 <= depth_atr <= 1.20:
            score += 0.20
        elif 0.20 <= depth_atr <= 1.60:
            score += 0.10

    # EMA25〜EMA50 付近で止まる（割り込み過ぎない）
    e25 = safe_float(ema25.iloc[-1], np.nan)
    e50 = safe_float(ema50.iloc[-1], np.nan)
    cl = safe_float(c.iloc[-1], np.nan)
    lo = safe_float(l.iloc[-1], np.nan)
    if np.isfinite(e25) and np.isfinite(e50) and np.isfinite(cl) and np.isfinite(lo):
        # 現在値・安値がEMA帯の近辺（下に抜け過ぎない）
        if lo >= e50 * 0.95:
            score += 0.12
        elif lo >= e50 * 0.92:
            score += 0.06
        if cl >= e25 * 0.98:
            score += 0.08

    # 出来高：ブレイク直後より調整局面で縮む（理想）
    vol_pre = float(np.nanmean(v.iloc[max(0, bidx-10):bidx])) if bidx >= 5 else float(np.nanmean(v.iloc[-20:]))
    vol_pb = float(np.nanmean(v.iloc[bidx:])) if bidx < len(df) else float(np.nanmean(v.iloc[-5:]))
    if np.isfinite(vol_pre) and np.isfinite(vol_pb) and vol_pre > 0:
        if vol_pb / vol_pre <= 0.85:
            score += 0.10
        elif vol_pb / vol_pre <= 1.00:
            score += 0.05

    # 下値切り上げ（HL）を簡易チェック：直近5本の安値が上向き
    l5 = pd.to_numeric(l.iloc[-5:], errors="coerce")
    if l5.isna().sum() == 0:
        if l5.iloc[-1] >= l5.iloc[0]:
            score += 0.08

    # 反転兆候：直近終値が前日高値を超える/陽線
    if len(df) >= 2:
        if safe_float(c.iloc[-1], 0.0) > safe_float(df["High"].iloc[-2], 0.0):
            score += 0.07
        elif safe_float(c.iloc[-1], 0.0) > safe_float(df["Open"].iloc[-1], 0.0):
            score += 0.04

    return float(clamp(score, 0.60, 1.40))

