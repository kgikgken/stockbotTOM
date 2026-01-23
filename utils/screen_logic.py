from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import ema, rsi14, atr14, adv20, clamp
from utils.rr_ev import rr as rr_calc, expected_days as exp_days_calc, turnover_efficiency


INDEX_TICKER = os.getenv("INDEX_TICKER", "1306.T")  # TOPIX ETF by default


@lru_cache(maxsize=1)
def _index_daily() -> pd.DataFrame:
    """Fetch index daily data once per run."""
    try:
        df = yf.download(INDEX_TICKER, period="6mo", interval="1d", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.dropna().copy()
    except Exception:
        return pd.DataFrame()


def load_universe(csv_path: str = "universe_jpx.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(
            [
                {"ticker": "2432.T", "name": "ディー・エヌ・エー", "sector": "情報・通信業"},
                {"ticker": "2216.T", "name": "カンロ", "sector": "食料品"},
                {"ticker": "7774.T", "name": "ジャパン・ティッシュエンジニアリング", "sector": "精密機器"},
            ]
        )
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("universe_jpx.csv must contain ticker column")
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    if "sector" not in df.columns:
        df["sector"] = "不明"
    return df[["ticker", "name", "sector"]].dropna().drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def fetch_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna()


def pullback_score(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 80:
        return 0.0

    close = df["Close"]
    high = df["High"]
    vol = df["Volume"]

    e25 = ema(close, 25)
    e50 = ema(close, 50)
    a14 = float(atr14(df).iloc[-1])
    if not np.isfinite(a14) or a14 <= 0:
        return 0.0

    up_regime = (e25.iloc[-1] > e50.iloc[-1]) and (close.iloc[-1] > e50.iloc[-1])
    if not up_regime:
        return 0.0

    last20 = high.tail(20)
    peak_idx = last20.idxmax()
    peak_pos = df.index.get_loc(peak_idx) if peak_idx in df.index else len(df) - 1
    bars_since_peak = (len(df) - 1) - peak_pos
    recent_break = high.iloc[-1] >= last20.max() * 0.999

    score = 40.0
    if recent_break:
        score += 15.0

    if 3 <= bars_since_peak <= 7:
        score += 20.0
    elif 1 <= bars_since_peak <= 10:
        score += 10.0
    else:
        score -= 5.0

    peak_high = float(high.iloc[peak_pos])
    depth = (peak_high - float(close.iloc[-1])) / a14
    if 0.4 <= depth <= 2.0:
        score += 15.0
    elif depth < 0.4:
        score += 5.0
    else:
        score -= 10.0

    v5 = float(vol.tail(5).mean())
    v10 = float(vol.tail(15).head(10).mean()) if len(vol) >= 15 else float(vol.tail(10).mean())
    if np.isfinite(v5) and np.isfinite(v10) and v10 > 0:
        if v5 <= v10 * 0.85:
            score += 10.0
        elif v5 <= v10 * 1.05:
            score += 5.0
        else:
            score -= 5.0

    rsi = float(rsi14(close).iloc[-1])
    if 48 <= rsi <= 75:
        score += 10.0
    elif rsi < 40:
        score -= 10.0

    return float(clamp(score, 0.0, 100.0))


def momentum_score(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 60:
        return 0.0
    close = df["Close"]
    rsi = float(rsi14(close).iloc[-1])
    e25 = ema(close, 25)
    slope25 = (e25.iloc[-1] / e25.iloc[-6] - 1.0) if np.isfinite(e25.iloc[-6]) else 0.0
    score = 50.0 + (rsi - 50.0) * 0.6 + clamp(slope25 * 2000.0, -10.0, 10.0)
    return float(clamp(score, 0.0, 100.0))


def breakout_score(df: pd.DataFrame) -> float:
    """Initial breakout quality score.

    Favours:
    - consolidation (low range expansion) over the last ~20-40 bars
    - clean break of a recent range high
    - meaningful volume expansion
    """
    if df is None or df.empty or len(df) < 80:
        return 0.0

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    vol = df["Volume"]

    # recent 20-bar range
    range_high = float(high.iloc[-21:-1].max())
    range_low = float(low.iloc[-21:-1].min())
    last_close = float(close.iloc[-1])
    if range_high <= 0 or range_low <= 0:
        return 0.0

    broke = 1.0 if last_close > range_high else 0.0
    if broke <= 0:
        return 0.0

    # consolidation: the tighter the 20-bar range vs price, the better
    range_pct = (range_high - range_low) / max(1e-9, range_high)
    tight = clamp(1.0 - (range_pct / 0.15), 0.0, 1.0)  # 15%+ range is "loose"

    vma = float(vol.rolling(20).mean().iloc[-1])
    vratio = float(vol.iloc[-1] / max(1.0, vma))
    v_boost = clamp((vratio - 1.0) / 1.5, 0.0, 1.0)  # 2.5x -> ~1.0

    # breakout distance: small is ok (we buy the retest later), huge is chase
    dist = (last_close - range_high) / max(1e-9, range_high)
    dist_ok = clamp(1.0 - (dist / 0.05), 0.0, 1.0)  # 5%+ is too stretched

    score = 40.0 + 25.0 * tight + 25.0 * v_boost + 10.0 * dist_ok
    return float(clamp(score, 0.0, 100.0))


def distortion_score(df: pd.DataFrame) -> float:
    """Supply/demand distortion score (exception engine).

    We approximate "distortion" as: relative strength vs index + reversal cue.
    It is intentionally sparse and conservative.
    """
    if df is None or df.empty or len(df) < 40:
        return 0.0

    idx = _index_daily()
    if idx is None or idx.empty:
        return 0.0

    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # align by dates
    common = close.index.intersection(idx.index)
    if len(common) < 20:
        return 0.0
    c = close.loc[common]
    ic = idx.loc[common, "Close"]

    r5 = (float(c.iloc[-1]) / float(c.iloc[-6]) - 1.0) if len(c) >= 6 else 0.0
    ir5 = (float(ic.iloc[-1]) / float(ic.iloc[-6]) - 1.0) if len(ic) >= 6 else 0.0
    rel = r5 - ir5
    rel_boost = clamp((rel + 0.03) / 0.06, 0.0, 1.0)  # +3% rel -> ~1.0

    # reversal candle: long lower wick + close > open (or strong close)
    last_o = float(open_.iloc[-1])
    last_c = float(close.iloc[-1])
    last_h = float(high.iloc[-1])
    last_l = float(low.iloc[-1])
    rng = max(1e-9, last_h - last_l)
    lower_wick = (min(last_o, last_c) - last_l) / rng
    body_pos = (last_c - last_o) / rng
    rev = clamp((lower_wick - 0.35) / 0.35, 0.0, 1.0) * clamp((body_pos + 0.05) / 0.25, 0.0, 1.0)

    # volume contraction after stress
    vma = float(vol.rolling(20).mean().iloc[-1])
    vratio = float(vol.iloc[-1] / max(1.0, vma))
    quiet = clamp(1.0 - (vratio / 1.2), 0.0, 1.0)

    score = 35.0 + 35.0 * rel_boost + 20.0 * rev + 10.0 * quiet
    return float(clamp(score, 0.0, 100.0))


def build_pullback_levels(df: pd.DataFrame) -> Dict:
    close = df["Close"]
    low = df["Low"]
    high = df["High"]
    a14 = float(atr14(df).iloc[-1])
    if not np.isfinite(a14) or a14 <= 0:
        return {}

    last = float(close.iloc[-1])
    e25 = float(ema(close, 25).iloc[-1])

    entry = (last * 0.6 + e25 * 0.4)
    sl_raw = float(low.tail(10).min())
    sl = sl_raw - 0.2 * a14

    tp1 = float(high.tail(20).max())
    tp2 = tp1 + 0.6 * (tp1 - entry)

    return {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "atr": a14}


def build_breakout_levels(df: pd.DataFrame) -> Dict:
    if df is None or df.empty or len(df) < 60:
        return {}
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    a14 = float(atr14(df).iloc[-1])
    if not np.isfinite(a14) or a14 <= 0:
        return {}
    # Breakout level: previous 20-bar high (excluding last bar)
    range_high = float(high.iloc[-21:-1].max())
    entry = range_high  # wait for retest; limit at the level
    sl = float(low.iloc[-10:].min()) - 0.2 * a14
    risk = max(1e-9, entry - sl)
    tp1 = entry + 1.2 * risk  # aim fast
    tp2 = entry + 1.8 * risk
    return {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "atr": a14}


def build_distortion_levels(df: pd.DataFrame) -> Dict:
    if df is None or df.empty or len(df) < 40:
        return {}
    close = df["Close"]
    low = df["Low"]
    a14 = float(atr14(df).iloc[-1])
    if not np.isfinite(a14) or a14 <= 0:
        return {}
    entry = float(close.iloc[-1])
    sl = float(low.iloc[-5:].min()) - 0.1 * a14
    risk = max(1e-9, entry - sl)
    # distortion is treated as small, short-lived edge: TP1 capped ~0.5R
    tp1 = entry + 0.5 * risk
    tp2 = entry + 0.8 * risk
    return {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "atr": a14}


# Backward-compatible alias
def build_trade_levels(df: pd.DataFrame) -> Dict:
    return build_pullback_levels(df)


def build_raw_candidates(universe: pd.DataFrame) -> Tuple[List[Dict], Dict]:
    raw: List[Dict] = []
    debug = {"raw": 0, "fetched": 0, "skipped": 0}

    for _, u in universe.iterrows():
        ticker = str(u["ticker"])
        name = str(u.get("name", ticker))
        sector = str(u.get("sector", "不明"))

        df = fetch_ohlcv(ticker)
        debug["raw"] += 1
        if df is None or df.empty or len(df) < 80:
            debug["skipped"] += 1
            continue
        debug["fetched"] += 1

        adv = adv20(df)
        pull = pullback_score(df)
        mom = momentum_score(df)
        brk = breakout_score(df)
        dist = distortion_score(df)

        pb_levels = build_pullback_levels(df)
        br_levels = build_breakout_levels(df)
        ds_levels = build_distortion_levels(df)

        if not pb_levels and not br_levels and not ds_levels:
            debug["skipped"] += 1
            continue

        raw.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "adv20": float(adv) if np.isfinite(adv) else float("nan"),
                "pullback_score": float(pull),
                "momentum_score": float(mom),
                "breakout_score": float(brk),
                "distortion_score": float(dist),
                "pb_levels": pb_levels,
                "br_levels": br_levels,
                "ds_levels": ds_levels,
            }
        )

    return raw, debug
