from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.rr import compute_tp_sl_rr


def _last(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return float('nan')


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return float('nan')
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else float('nan')


def _adv20(df: pd.DataFrame) -> float:
    c = df['Close'].astype(float)
    v = df['Volume'].astype(float) if 'Volume' in df.columns else pd.Series(0.0, index=df.index)
    adv = (c * v).rolling(20).mean().iloc[-1]
    return float(adv) if np.isfinite(adv) else float('nan')


def _rsi14(close: pd.Series) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return _last(rsi)


def _sma(close: pd.Series, n: int) -> float:
    if close is None or len(close) < n:
        return _last(close)
    return float(close.rolling(n).mean().iloc[-1])


def _slope_sma(close: pd.Series, n: int, lookback: int = 5) -> float:
    if close is None or len(close) < n + lookback:
        return float('nan')
    ma = close.rolling(n).mean()
    a = float(ma.iloc[-1])
    b = float(ma.iloc[-1 - lookback])
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return float('nan')
    return (a / b - 1.0)


# ------------------------------------------------------------
# TrendGate（逆張り排除）
# ------------------------------------------------------------
def trend_gate(hist: pd.DataFrame) -> bool:
    if hist is None or len(hist) < 80:
        return False
    c = hist['Close'].astype(float)
    last = _last(c)
    ma20 = _sma(c, 20)
    ma50 = _sma(c, 50)
    if not (np.isfinite(last) and np.isfinite(ma20) and np.isfinite(ma50)):
        return False
    if not (last > ma20 > ma50):
        return False
    slope20 = _slope_sma(c, 20, 5)
    if not np.isfinite(slope20) or slope20 <= 0:
        return False
    return True


# ------------------------------------------------------------
# Universeフィルタ
# ------------------------------------------------------------
def compute_universe_filters(
    hist: pd.DataFrame,
    price_min: float,
    price_max: float,
    adv20_min: float,
    atr_pct_min: float,
) -> Tuple[bool, Dict]:
    c = hist['Close'].astype(float)
    price = _last(c)
    if not np.isfinite(price) or price <= 0:
        return False, {}
    if price < price_min or price > price_max:
        return False, {}
    adv = _adv20(hist)
    if not np.isfinite(adv) or adv < adv20_min:
        return False, {}
    atr = _atr(hist, 14)
    if not np.isfinite(atr) or atr <= 0:
        return False, {}
    atr_pct = atr / price
    if atr_pct < atr_pct_min:
        return False, {}
    return True, {'adv20': float(adv), 'atr_pct': float(atr_pct)}


# ------------------------------------------------------------
# Setup検出（A:押し目 / B:ブレイク）
# ------------------------------------------------------------
def detect_setup_type(hist: pd.DataFrame) -> str:
    if hist is None or len(hist) < 80:
        return 'N'
    df = hist.copy()
    c = df['Close'].astype(float)
    v = df['Volume'].astype(float) if 'Volume' in df.columns else pd.Series(0.0, index=df.index)
    last = _last(c)
    atr = _atr(df, 14)
    if not np.isfinite(last) or not np.isfinite(atr) or atr <= 0:
        return 'N'
    ma20 = _sma(c, 20)
    ma50 = _sma(c, 50)
    slope20 = _slope_sma(c, 20, 5)

    # A: トレンド押し目
    if np.isfinite(ma20) and np.isfinite(ma50) and np.isfinite(slope20):
        if last > ma20 > ma50 and slope20 > 0:
            if abs(last - ma20) <= 0.8 * atr:
                return 'A'

    # B: ブレイク
    if len(c) >= 21:
        hh20 = float(c.tail(21).iloc[:-1].max())
        vol_ma20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float('nan')
        if np.isfinite(hh20) and last > hh20:
            if np.isfinite(vol_ma20) and vol_ma20 > 0 and float(v.iloc[-1]) >= 1.5 * vol_ma20:
                return 'B'
    return 'N'


# ------------------------------------------------------------
# INゾーン
# ------------------------------------------------------------
def calc_in_zone(hist: pd.DataFrame, setup: str) -> Tuple[float, float, float, float]:
    df = hist.copy()
    c = df['Close'].astype(float)
    atr = _atr(df, 14)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(_last(c) * 0.01, 1.0)

    if setup == 'A':
        ma20 = _sma(c, 20)
        center = ma20
        low = center - 0.5 * atr
        high = center + 0.5 * atr
        return float(low), float(high), float(center), float(atr)

    if len(c) >= 21:
        hh20 = float(c.tail(21).iloc[:-1].max())
    else:
        hh20 = _last(c)
    center = hh20
    low = center - 0.3 * atr
    high = center + 0.3 * atr
    return float(low), float(high), float(center), float(atr)


# ------------------------------------------------------------
# Pwin推定（代理特徴）
# ------------------------------------------------------------
def estimate_pwin(hist: pd.DataFrame, sector_rank: int) -> float:
    df = hist.copy()
    c = df['Close'].astype(float)
    last = _last(c)
    ma20 = _sma(c, 20)
    ma50 = _sma(c, 50)
    slope20 = _slope_sma(c, 20, 5)
    rsi = _rsi14(c)
    adv = _adv20(df)

    sc = 0.0

    # TrendStrength
    if np.isfinite(last) and np.isfinite(ma20) and np.isfinite(ma50):
        if last > ma20 > ma50:
            sc += 0.20
        elif last > ma20:
            sc += 0.12
    if np.isfinite(slope20):
        sc += float(np.clip(slope20 / 0.02, 0.0, 0.15))

    # RSI
    if np.isfinite(rsi):
        if 45 <= rsi <= 60:
            sc += 0.20
        elif 40 <= rsi < 45 or 60 < rsi <= 68:
            sc += 0.12
        else:
            sc += 0.06

    # SectorRank
    sc += float(np.clip(0.18 - 0.03 * sector_rank, 0.03, 0.15))

    # Liquidity
    if np.isfinite(adv):
        if adv >= 1e9:
            sc += 0.10
        elif adv >= 2e8:
            sc += 0.06
        elif adv >= 1e8:
            sc += 0.04

    p = 0.15 + sc
    return float(np.clip(p, 0.10, 0.70))


def regime_multiplier(mkt_score: int, d_mkt_3d: int, events_soon: bool) -> float:
    mul = 1.0
    if mkt_score >= 60 and d_mkt_3d >= 0:
        mul *= 1.05
    if d_mkt_3d <= -5:
        mul *= 0.70
    if events_soon:
        mul *= 0.75
    return float(np.clip(mul, 0.60, 1.10))


def calc_expected_days(rr_info: Dict, atr: float) -> float:
    entry = float(rr_info.get('entry', 0.0))
    tp2 = float(rr_info.get('tp_price', 0.0))
    if not (entry > 0 and tp2 > entry and atr > 0):
        return 9.9
    k = 1.0
    return float((tp2 - entry) / (k * atr))


# ------------------------------------------------------------
# 候補メトリクス（RR/EV/AdjEV/速度/GU）
# ------------------------------------------------------------
def compute_candidate_metrics(
    hist: pd.DataFrame,
    setup: str,
    sector_rank: int,
    mkt_score: int,
    d_mkt_3d: int,
    events_soon: bool,
) -> Optional[Dict]:
    rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score, for_day=False)
    rr = float(rr_info.get('rr', 0.0))
    if rr < 2.2:
        return None

    pwin = estimate_pwin(hist, sector_rank=sector_rank)
    ev = pwin * rr - (1.0 - pwin) * 1.0
    if ev < 0.40:
        return None

    atr = _atr(hist, 14)
    if not np.isfinite(atr) or atr <= 0:
        return None

    exp_days = calc_expected_days(rr_info, atr)
    if exp_days > 5.0:
        return None

    rpd = rr / exp_days if exp_days > 0 else 0.0
    if rpd < 0.50:
        return None

    close = hist['Close'].astype(float)
    open_ = hist['Open'].astype(float)
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else float('nan')
    open_last = _last(open_)
    gu_flag = bool(np.isfinite(open_last) and np.isfinite(prev_close) and open_last > (prev_close + 1.0 * atr))

    mul = regime_multiplier(mkt_score=mkt_score, d_mkt_3d=d_mkt_3d, events_soon=events_soon)
    adj_ev = ev * mul
    if adj_ev < 0.40:
        return None

    entry = float(rr_info['entry'])
    stop = float(rr_info['sl_price'])
    r_unit = entry - stop
    tp1 = entry + 1.5 * r_unit
    tp2 = float(rr_info['tp_price'])

    return dict(
        rr=float(rr),
        pwin=float(pwin),
        ev=float(ev),
        adj_ev=float(adj_ev),
        expected_days=float(exp_days),
        r_per_day=float(rpd),
        gu_flag=bool(gu_flag),
        entry=float(entry),
        stop=float(stop),
        tp1=float(tp1),
        tp2=float(tp2),
    )


def decide_action(hist: pd.DataFrame, in_center: float, in_low: float, in_high: float, atr: float) -> str:
    price_now = float(hist['Close'].iloc[-1])
    if not (np.isfinite(price_now) and np.isfinite(in_center) and np.isfinite(atr) and atr > 0):
        return '監視のみ'

    dist = abs(price_now - in_center) / atr
    if dist > 0.8:
        return '指値待ち'
    if price_now > in_high or price_now < in_low:
        return '指値待ち'
    return '即IN可'


# ------------------------------------------------------------
# 選抜（AdjEV降順 → セクター上限 → GU/指値待ちは監視へ）
# ------------------------------------------------------------
def build_portfolio_selection(
    candidates: List[Dict],
    max_final: int,
    watch_max: int,
    max_per_sector: int,
    corr_lookback: int,
    corr_max: float,
) -> Tuple[List[Dict], List[Dict]]:
    if not candidates:
        return [], []

    cands = sorted(candidates, key=lambda x: (x['adj_ev'], x['r_per_day']), reverse=True)

    selected: List[Dict] = []
    watch: List[Dict] = []
    sector_counts: Dict[str, int] = {}

    for c in cands:
        sec = str(c.get('sector', '不明'))
        sector_counts.setdefault(sec, 0)

        if c.get('gu_flag') or c.get('action') == '指値待ち':
            if len(watch) < watch_max:
                c2 = dict(c)
                c2['watch_reason'] = 'GU/追いかけ禁止'
                watch.append(c2)
            continue

        if sector_counts[sec] >= max_per_sector:
            if len(watch) < watch_max:
                c2 = dict(c)
                c2['watch_reason'] = 'セクター上限'
                watch.append(c2)
            continue

        selected.append(c)
        sector_counts[sec] += 1

        if len(selected) >= max_final:
            break

    if len(watch) < watch_max:
        for c in cands:
            if any(x['ticker'] == c['ticker'] for x in selected) or any(x['ticker'] == c['ticker'] for x in watch):
                continue
            if len(watch) >= watch_max:
                break
            c2 = dict(c)
            c2['watch_reason'] = '候補'
            watch.append(c2)

    return selected, watch
