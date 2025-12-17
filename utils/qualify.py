from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _sma(s: pd.Series, n: int) -> float:
    if s is None or len(s) < n:
        return _last(s)
    return float(s.rolling(n).mean().iloc[-1])


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan


def qualify_runner(hist: pd.DataFrame) -> Dict[str, float | str]:
    """
    Runner分類（A1/A2_prebreak/B/C）と走行強度(0-100)
    vAB_prime++: AL3の前提は A2_prebreak のみ。

    A1: すでに走ってしまった（追い禁止）
    A2_prebreak: 走る準備（押し戻し→再上昇の余地）
    B/C: 除外寄り
    """
    if hist is None or len(hist) < 80:
        return {"runner": "C", "strength": 0.0}

    df = hist.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    c = _last(close)
    if not np.isfinite(c) or c <= 0:
        return {"runner": "C", "strength": 0.0}

    ma20 = _sma(close, 20)
    ma60 = _sma(close, 60)

    atr = _atr(df, 14)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(c * 0.01, 1.0)

    trend_ok = bool(np.isfinite(ma20) and np.isfinite(ma60) and c > ma20 > ma60)

    # 直近10日レンジの位置
    hi10 = float(high.tail(10).max())
    lo10 = float(low.tail(10).min())
    rng = max(hi10 - lo10, 1e-9)
    pos = float((c - lo10) / rng)  # 0..1

    # 直近3日での“走り”強さ（ATR比）
    surge = float(close.iloc[-1] - close.iloc[-4]) if len(close) >= 4 else 0.0
    surge_atr = surge / atr

    # 押し戻しがあるか（直近高値から0.4〜1.2ATR）
    recent_high = float(high.iloc[-6:-1].max()) if len(high) >= 6 else float(high.max())
    pullback = float(recent_high - c)
    pullback_atr = pullback / atr
    pullback_ok = bool(0.4 <= pullback_atr <= 1.2)

    # 分類
    if trend_ok and surge_atr >= 1.8 and pos >= 0.80:
        runner = "A1"
    elif trend_ok and surge_atr < 1.8 and pullback_ok and pos >= 0.45:
        runner = "A2_prebreak"
    elif trend_ok and pos >= 0.35:
        runner = "B"
    else:
        runner = "C"

    # 強度 0-100
    sc = 0.0
    if trend_ok:
        sc += 35.0

    # 走り過ぎでないほど加点（A1は別途弾く）
    sc += float(np.clip((1.8 - max(surge_atr, 0.0)) / 1.8, 0, 1)) * 20.0

    if pullback_ok:
        sc += 25.0
    else:
        sc += float(np.clip(1.0 - abs(pullback_atr - 0.8) / 1.2, 0, 1)) * 12.0

    sc += float(np.clip(pos, 0, 1)) * 20.0

    strength = float(np.clip(sc, 0, 100))
    return {"runner": runner, "strength": strength}


def calc_al(runner: str, strength: float, in_rank: str, rr: float, ev_r: float) -> int:
    """
    vAB_prime++:
      - AL3の前提は A2_prebreak かつ strength>=70
      - A1は追い禁止なのでAL1扱い
      - B/CはAL1（Swing枠に出さない）
    """
    runner = (runner or "").strip()

    if runner == "A1":
        return 1

    if runner == "A2_prebreak":
        if strength >= 70 and in_rank in ("強IN", "通常IN") and rr >= 2.0 and ev_r >= 0.45:
            return 3
        return 1

    return 1


def al3_score(runner_strength: float, ev_r: float, rr: float, gap_pct: float, in_rank: str) -> float:
    """
    AL3が複数出た場合の一位決定用。

    PullbackQuality:
      - 現在が押し目基準より上に飛んでる（gapが大きい）ほど減点
      - 目安: gap 0〜1.5% 良し、>3% 減点、>5% 強い減点
    """
    g = float(gap_pct) if np.isfinite(gap_pct) else 0.0

    if g <= 1.5:
        pull_q = 1.0
    elif g <= 3.0:
        pull_q = 0.9
    elif g <= 5.0:
        pull_q = 0.75
    else:
        pull_q = 0.60

    if in_rank == "強IN":
        in_w = 1.05
    elif in_rank == "通常IN":
        in_w = 1.00
    else:
        in_w = 0.92

    rs = float(runner_strength) / 100.0
    return float(rs * float(ev_r) * float(rr) * float(pull_q) * float(in_w))
