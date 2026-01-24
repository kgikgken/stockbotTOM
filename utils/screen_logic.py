from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import ema, rsi14, atr14, adv20, clamp
from utils.rr_ev import rr as rr_calc, expected_days as exp_days_calc, turnover_efficiency


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


def build_trade_levels(df: pd.DataFrame) -> Dict:
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
        levels = build_trade_levels(df)
        if not levels:
            debug["skipped"] += 1
            continue

        entry = levels["entry"]
        sl = levels["sl"]
        tp1 = levels["tp1"]
        tp2 = levels["tp2"]
        atr = levels["atr"]

        rr2 = rr_calc(entry, sl, tp2)
        if not np.isfinite(rr2):
            debug["skipped"] += 1
            continue

        expd = exp_days_calc(entry, tp2, atr)
        rday = turnover_efficiency(rr2, expd)

        struct_ev = (rr2 / 2.0) * (0.5 + pull / 200.0) * (0.8 + mom / 250.0)
        struct_ev = float(clamp(struct_ev, -1.0, 3.0))

        raw.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "adv20": float(adv) if np.isfinite(adv) else float("nan"),
                "pullback_score": float(pull),
                "momentum_score": float(mom),
                "entry": float(entry),
                "sl": float(sl),
                "tp1": float(tp1),
                "tp2": float(tp2),
                "rr": float(rr2),
                "expected_days": float(expd) if np.isfinite(expd) else float("nan"),
                "rday": float(rday) if np.isfinite(rday) else float("nan"),
                "struct_ev": float(struct_ev),
            }
        )

    return raw, debug
