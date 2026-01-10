from __future__ import annotations

from typing import List, Dict, Tuple
import time
import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import business_days_diff
from utils.setup import detect_setup
from utils.rr_ev import compute_exit_levels, compute_ev_metrics
from utils.market import rr_min_for_market

EARNINGS_EXCLUDE_BD = 3

def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return float(v) if np.isfinite(v) else float(default)
    except Exception:
        return float(default)

def _fetch_hist(ticker: str, period: str = "260d") -> pd.DataFrame | None:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty and len(df) >= 80:
                return df
        except Exception:
            time.sleep(0.4)
    return None

def _atr_pct(atr: float, price: float) -> float:
    if not (np.isfinite(atr) and np.isfinite(price) and price > 0):
        return 0.0
    return float(atr / price * 100.0)

def _adv20_jpy(df: pd.DataFrame) -> float:
    try:
        close = df["Close"].astype(float)
        vol = df["Volume"].astype(float)
        adv = (close * vol).rolling(20).mean().iloc[-1]
        return float(adv) if np.isfinite(adv) else 0.0
    except Exception:
        return 0.0

def _exclude_abnormal(df: pd.DataFrame) -> bool:
    # Exclude extreme moves or locked limit (approx)
    try:
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        ret = close.pct_change(fill_method=None).tail(5).abs()
        if float(ret.max()) > 0.18:
            return True
        # lock candles: high==low for 2+ days with tiny range
        rng = (high - low).tail(3)
        if (rng < (close.tail(3) * 0.001)).all():
            return True
    except Exception:
        return False
    return False

def _earnings_block(row: pd.Series, today_date) -> bool:
    # returns True if blocked
    if "earnings_date" not in row.index:
        return False
    ed = str(row.get("earnings_date", "")).strip()
    if not ed:
        return False
    try:
        d = pd.to_datetime(ed, errors="coerce").date()
    except Exception:
        return False
    if d is None or pd.isna(d):
        return False
    bd = abs(business_days_diff(today_date, d))
    return bool(bd <= EARNINGS_EXCLUDE_BD)

def build_raw_candidates(universe: pd.DataFrame, today_date, mkt_score: int, macro_on: bool) -> Tuple[List[Dict], Dict]:
    rr_min = rr_min_for_market(mkt_score)
    out: List[Dict] = []
    stats = {"raw_n": 0, "filtered_n": 0}

    # ticker column
    if "ticker" in universe.columns:
        t_col = "ticker"
    elif "code" in universe.columns:
        t_col = "code"
    else:
        return [], {"raw_n": 0, "filtered_n": 0}

    for _, row in universe.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue
        stats["raw_n"] += 1

        # earnings block
        if _earnings_block(row, today_date):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = _fetch_hist(ticker, period="260d")
        if hist is None:
            continue
        if _exclude_abnormal(hist):
            continue

        price = _safe_float(hist["Close"].iloc[-1], np.nan)
        if not (np.isfinite(price) and 200 <= price <= 15000):
            continue

        setup, anchors, gu = detect_setup(hist)
        if setup == "NONE":
            continue

        atr = float(anchors.get("atr", np.nan))
        atrp = _atr_pct(atr, price)
        if atrp < 1.5:
            continue

        adv = _adv20_jpy(hist)
        if adv < 200_000_000:
            continue

        exits = compute_exit_levels(hist, anchors["entry_mid"], atr)
        rr = float(exits["rr"])
        if rr < rr_min:
            continue

        ev, adjev, expected_days, rday = compute_ev_metrics(
            setup=setup,
            rr=rr,
            atr=atr,
            entry=anchors["entry_mid"],
            tp2=exits["tp2"],
            mkt_score=mkt_score,
            macro_on=macro_on,
            gu=gu,
        )

        if adjev < 0.50:
            continue
        if rday < 0.50:
            continue

        # action
        action = "監視（寄り後再判定）" if gu else "指値（Entry帯で待つ）"

        # returns for correlation (60d)
        try:
            ret60 = hist["Close"].astype(float).pct_change(fill_method=None).tail(61)
        except Exception:
            ret60 = None

        out.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "setup": setup,
            "action": action,
            "entry_lo": float(anchors["entry_lo"]),
            "entry_hi": float(anchors["entry_hi"]),
            "rr": float(rr),
            "ev": float(ev),
            "adjev": float(adjev),
            "expected_days": float(expected_days),
            "rday": float(rday),
            "sl": float(exits["sl"]),
            "tp1": float(exits["tp1"]),
            "tp2": float(exits["tp2"]),
            "gu": bool(gu),
            "_ret60": ret60,
            "adv20": float(adv),
            "atrp": float(atrp),
        })

    stats["filtered_n"] = len(out)
    return out, stats
