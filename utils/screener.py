from __future__ import annotations

import os
import time
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from utils.screen_logic import detect_setup, score_candidate
from utils.diversify import diversify

PRICE_MIN = 200
PRICE_MAX = 15000
ADV20_MIN = 200_000_000
ATR_PCT_MIN = 1.5
EARN_EXCLUDE_DDAYS = 3

MAX_LINE = 5
MAX_LINE_MACRO = 2
GU_RATIO_MAX = 0.60

def rr_min_by_market(mkt_score: int) -> float:
    if mkt_score >= 70:
        return 1.8
    if mkt_score >= 60:
        return 2.0
    if mkt_score >= 50:
        return 2.2
    return 2.5

def _fetch_history(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty and len(df) >= 120:
                return df
        except Exception:
            time.sleep(0.25)
    return None

def _earnings_block(earn_date_str: str, today_date) -> bool:
    if not earn_date_str or str(earn_date_str).strip() == "" or str(earn_date_str).lower() == "nan":
        return False
    try:
        d = pd.to_datetime(earn_date_str, errors="coerce").date()
        if d is None:
            return False
        delta = abs((d - today_date).days)
        return delta <= EARN_EXCLUDE_DDAYS
    except Exception:
        return False

def _abnormal_filter(hist: pd.DataFrame) -> bool:
    try:
        close = hist["Close"].astype(float)
        ret = close.pct_change().dropna()
        if len(ret) < 5:
            return False
        spikes = int((ret.abs() >= 0.18).sum())
        return spikes >= 2
    except Exception:
        return False

def run_screening(
    universe_path: str,
    today_date,
    mkt: Dict[str, Any],
    macro_on: bool,
    futures_risk_on: bool,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    mkt_score = int(mkt.get("score", 50))
    rr_min = rr_min_by_market(mkt_score)

    delta3d = float(state.get("delta3d", 0.0) or 0.0)
    if mkt_score < 45 or (delta3d <= -5 and mkt_score < 55):
        return {"no_trade": True, "reason": "地合いNG", "candidates": [], "meta": {"rr_min": rr_min}}

    weekly_new = int(state.get("weekly_new", 0) or 0)
    if weekly_new >= 3:
        return {"no_trade": True, "reason": "週次新規上限", "candidates": [], "meta": {"rr_min": rr_min}}

    if not os.path.exists(universe_path):
        return {"no_trade": True, "reason": "universe_jpx.csvが見つからない", "candidates": [], "meta": {"rr_min": rr_min}}

    try:
        uni = pd.read_csv(universe_path)
    except Exception:
        return {"no_trade": True, "reason": "universe読み込み失敗", "candidates": [], "meta": {"rr_min": rr_min}}

    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return {"no_trade": True, "reason": "ticker列なし", "candidates": [], "meta": {"rr_min": rr_min}}

    if "sector" in uni.columns:
        s_col = "sector"
    elif "industry_big" in uni.columns:
        s_col = "industry_big"
    else:
        s_col = None

    cands: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        if _earnings_block(str(row.get("earnings_date", "")).strip(), today_date):
            continue

        hist = _fetch_history(ticker)
        if hist is None:
            continue

        try:
            close = hist["Close"].astype(float)
            price = float(close.iloc[-1])
            if not (PRICE_MIN <= price <= PRICE_MAX):
                continue
        except Exception:
            continue

        if _abnormal_filter(hist):
            continue

        try:
            vol = hist["Volume"].astype(float)
            adv20 = float((close * vol).rolling(20).mean().iloc[-1])
            if not (np.isfinite(adv20) and adv20 >= ADV20_MIN):
                continue
        except Exception:
            continue

        try:
            high = hist["High"].astype(float)
            low = hist["Low"].astype(float)
            prev_close = close.shift(1)
            tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-1])
            atr_pct = float(atr14 / price * 100.0) if price > 0 else 0.0
            if not (np.isfinite(atr_pct) and atr_pct >= ATR_PCT_MIN):
                continue
        except Exception:
            continue

        setup, anchors = detect_setup(hist)
        if setup == "NA":
            continue

        scored = score_candidate(hist, setup, anchors, mkt_score=mkt_score, macro_on=macro_on)
        if not scored:
            continue

        if float(scored["rr"]) < rr_min:
            continue
        if float(scored["adjev"]) < 0.50:
            continue
        if float(scored["rday"]) < 0.50:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get(s_col, "不明")) if s_col else "不明"

        c = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "setup": setup,
            "entry_lo": scored["entry_lo"],
            "entry_hi": scored["entry_hi"],
            "rr": scored["rr"],
            "adjev": scored["adjev"],
            "rday": scored["rday"],
            "expected_days": scored["expected_days"],
            "sl": scored["sl"],
            "tp1": scored["tp1"],
            "tp2": scored["tp2"],
            "gu": scored["gu"],
            "action": "寄り後再判定（GU）" if scored["gu"] else ("指値（ロット50%・TP2控えめ）" if macro_on else "指値"),
            "_close_series": hist["Close"].astype(float).tail(90),
        }
        cands.append(c)

    cands.sort(key=lambda x: (float(x["adjev"]), float(x["rday"]), float(x["rr"])), reverse=True)

    diversified = diversify(cands[:120], sector_cap=2, corr_max=0.75)

    cap = MAX_LINE if (macro_on and futures_risk_on) else (MAX_LINE_MACRO if macro_on else MAX_LINE)
    candidates = diversified[:cap]

    gu_ratio = float(np.mean([1.0 if c.get("gu", False) else 0.0 for c in candidates])) if candidates else 0.0
    if gu_ratio > GU_RATIO_MAX:
        candidates = [c for c in candidates if not bool(c.get("gu", False))][:cap]
        gu_ratio = float(np.mean([1.0 if c.get("gu", False) else 0.0 for c in candidates])) if candidates else 0.0

    if not candidates:
        return {"no_trade": True, "reason": "条件未達（候補ゼロ）", "candidates": [], "meta": {"rr_min": rr_min}}

    return {"no_trade": False, "reason": "", "candidates": candidates, "meta": {"rr_min": rr_min}}
