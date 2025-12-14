from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd

from utils.scoring import calc_inout_for_stock
from utils.rr import compute_tp_sl_rr


def _last(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _ma(close: pd.Series, n: int) -> float:
    if close is None or len(close) < n:
        return _last(close)
    return float(close.rolling(n).mean().iloc[-1])


def _rsi14(close: pd.Series) -> float:
    if close is None or len(close) < 20:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return _last(rsi)


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan


def _atr60(df: pd.DataFrame, atr_period: int = 14) -> float:
    # ATR14を作って60日平均
    if df is None or len(df) < atr_period + 60 + 5:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(atr_period).mean()
    v = atr14.rolling(60).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan


def _std(close: pd.Series, n: int) -> float:
    if close is None or len(close) < n + 2:
        return np.nan
    r = close.pct_change(fill_method=None)
    v = r.rolling(n).std().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan


def _days_since_rolling_high(close: pd.Series, window: int = 60) -> Optional[int]:
    if close is None or len(close) < window + 5:
        return None
    sub = close.tail(window)
    hi = float(sub.max())
    if not np.isfinite(hi) or hi <= 0:
        return None
    idx = sub[sub >= hi * 0.999999].index
    if len(idx) == 0:
        return None
    pos = list(sub.index).index(idx[-1])
    return (len(sub) - 1) - pos


def _pivot_high_count(close: pd.Series, lo: float, hi: float, window: int = 5) -> int:
    if close is None or len(close) < window + 10:
        return 999
    x = close.astype(float).values
    n = len(x)
    half = window // 2
    cnt = 0
    for i in range(half, n - half):
        c = x[i]
        if not np.isfinite(c):
            continue
        if c < lo or c > hi:
            continue
        seg = x[i - half : i + half + 1]
        if np.all(np.isfinite(seg)) and c == np.max(seg):
            cnt += 1
    return int(cnt)


def _time_pullback_ok(df: pd.DataFrame, atr14: float) -> bool:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    if len(close) < 12 or not np.isfinite(atr14) or atr14 <= 0:
        return False

    # 3〜7日“横〜微下げ”の近似：7日変化が+0.8%以内
    last7 = close.tail(7)
    chg7 = float(last7.iloc[-1] / last7.iloc[0] - 1.0)
    if chg7 > 0.008:
        return False

    # 直近5日の平均レンジが過大なら除外
    rng5 = (high.tail(5) - low.tail(5)).astype(float)
    rng5_mean = float(rng5.mean()) if len(rng5) == 5 else np.nan
    if np.isfinite(rng5_mean) and rng5_mean > atr14 * 1.2:
        return False

    # 押し戻し 0.3〜1.2ATR
    recent_high = float(high.iloc[-6:-1].max())
    c = _last(close)
    pull = float(recent_high - c)
    pull_atr = pull / atr14 if atr14 > 0 else 999.0
    if not (0.3 <= pull_atr <= 1.2):
        return False

    return True


def _volume_dry_ok(df: pd.DataFrame) -> bool:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else None
    if vol is None or len(vol) < 25:
        return False
    turnover = close * vol
    t3 = float(turnover.tail(3).mean())
    t20 = float(turnover.tail(20).mean())
    return bool(np.isfinite(t3) and np.isfinite(t20) and t20 > 0 and t3 < t20)


def expected_r_from_in_rank(in_rank: str, rr: float) -> float:
    if rr <= 0:
        return -999.0
    if in_rank == "強IN":
        win = 0.45
    elif in_rank == "通常IN":
        win = 0.40
    elif in_rank == "弱めIN":
        win = 0.33
    else:
        win = 0.25
    lose = 1.0 - win
    return float(win * rr - lose * 1.0)


@dataclass
class SwingPick:
    ticker: str
    name: str
    sector: str
    score: float
    al: int
    lev: float
    in_rank: str
    rr: float
    ev_r: float
    resistance: int
    entry: float
    price_now: float
    gap_pct: float
    tp_pct: float
    sl_pct: float
    tp_price: float
    sl_price: float
    trail_pct: float
    trail_price: float


def qualify_swing(hist: pd.DataFrame, mkt_score: int) -> Tuple[bool, str, Dict]:
    """
    vAB Swingフィルタ
    戻り：(ok, reason, payload)
    """
    if hist is None or len(hist) < 220:
        return False, "data_short", {}

    df = hist.copy()
    close = df["Close"].astype(float)
    price = _last(close)
    if not np.isfinite(price) or price <= 0:
        return False, "bad_price", {}

    ma20 = _ma(close, 20)
    ma60 = _ma(close, 60)

    # A-1 trend
    if not (np.isfinite(ma20) and np.isfinite(ma60) and price > ma20 > ma60):
        return False, "trend_fail", {}

    # MA60 slope (5日前比較)
    try:
        ma60_prev = float(close.rolling(60).mean().iloc[-6])
    except Exception:
        ma60_prev = np.nan
    if not (np.isfinite(ma60_prev) and ma60_prev > 0 and ma60 > ma60_prev):
        return False, "ma60_slope_fail", {}

    # 60d high recent (<=40日)
    dsh = _days_since_rolling_high(close, window=60)
    if dsh is None or dsh > 40:
        return False, "no_recent_breakout", {}

    # A-2 contraction
    atr14 = _atr(df, 14)
    atr60 = _atr60(df, 14)
    if not (np.isfinite(atr14) and np.isfinite(atr60) and atr14 > 0 and atr60 > 0 and atr14 < atr60 * 0.90):
        return False, "atr_contraction_fail", {}

    std20 = _std(close, 20)
    std60 = _std(close, 60)
    if not (np.isfinite(std20) and np.isfinite(std60) and std20 < std60 * 0.90):
        return False, "std_contraction_fail", {}

    # A-3 resistance
    hi180 = float(close.tail(180).max())
    lo = price * 1.01
    hi = hi180 * 0.995
    resistance = _pivot_high_count(close.tail(220), lo=lo, hi=hi, window=5)
    if resistance > 1:
        return False, "resistance_many", {"resistance": resistance}

    # B-1 time pullback
    if not _time_pullback_ok(df, atr14):
        return False, "pullback_fail", {}

    # B-2 volume dry
    if not _volume_dry_ok(df):
        return False, "volume_not_dry", {}

    # B-3 RSI
    rsi = _rsi14(close)
    if not (np.isfinite(rsi) and 28.0 <= rsi <= 62.0):
        return False, "rsi_reject", {"rsi": rsi}

    # IN rank（既存）
    in_rank, _, _ = calc_inout_for_stock(df)
    if in_rank == "様子見":
        return False, "in_rank_fail", {}

    rr_info = compute_tp_sl_rr(df, mkt_score=mkt_score, for_day=False)
    rr = float(rr_info.get("rr", 0.0))
    entry = float(rr_info.get("entry", 0.0))

    if not (np.isfinite(rr) and rr > 0 and np.isfinite(entry) and entry > 0):
        return False, "rr_fail", {}

    price_now = float(price)
    gap_pct = (price_now / entry - 1.0) * 100.0 if entry > 0 else 999.0

    # 追い禁止
    if np.isfinite(gap_pct) and gap_pct > 1.5:
        return False, "chase_block", {"gap_pct": gap_pct}

    ev_r = expected_r_from_in_rank(in_rank, rr)

    # AL判定
    if rr >= 2.5 and ev_r >= 0.6 and resistance <= 1 and gap_pct <= 1.0:
        al = 3
    elif rr >= 2.0 and ev_r >= 0.4 and resistance <= 2 and gap_pct <= 1.2:
        al = 2
    elif rr >= 1.8 and ev_r >= 0.3 and gap_pct <= 1.5:
        al = 1
    else:
        return False, "threshold_fail", {"rr": rr, "ev_r": ev_r, "resistance": resistance, "gap_pct": gap_pct}

    # TRAIL（大勝ち用）
    trail_pct = max(2.0 * (atr14 / price_now), 0.03) if np.isfinite(atr14) and atr14 > 0 else 0.03
    trail_price = price_now * (1.0 - trail_pct)

    payload = dict(
        al=int(al),
        in_rank=str(in_rank),
        rr=float(rr),
        ev_r=float(ev_r),
        resistance=int(resistance),
        entry=float(entry),
        price_now=float(price_now),
        gap_pct=float(gap_pct),
        tp_pct=float(rr_info.get("tp_pct", 0.0)),
        sl_pct=float(rr_info.get("sl_pct", 0.0)),
        tp_price=float(rr_info.get("tp_price", 0.0)),
        sl_price=float(rr_info.get("sl_price", 0.0)),
        trail_pct=float(trail_pct),
        trail_price=float(trail_price),
    )
    return True, "ok", payload


def day_event_ok(hist_d: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Dayの“事件”条件（事前判定）
    - value1 >= 1.5*value20
    - ATR14/ATR60 >= 1.10
    """
    if hist_d is None or len(hist_d) < 120:
        return False, {}
    close = hist_d["Close"].astype(float)
    vol = hist_d["Volume"].astype(float) if "Volume" in hist_d.columns else None
    if vol is None:
        return False, {}

    turnover = close * vol
    value1 = float(turnover.iloc[-1])
    value20 = float(turnover.tail(20).mean())
    if not (np.isfinite(value1) and np.isfinite(value20) and value20 > 0):
        return False, {}

    atr14 = _atr(hist_d, 14)
    atr60 = _atr60(hist_d, 14)
    if not (np.isfinite(atr14) and np.isfinite(atr60) and atr60 > 0):
        return False, {}

    ok = bool(value1 >= 1.5 * value20 and (atr14 / atr60) >= 1.10)
    return ok, {"value1": value1, "value20": value20, "atr_ratio": float(atr14/atr60)}
