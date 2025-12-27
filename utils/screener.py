from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.events import is_major_event_tomorrow
from utils.features import compute_features
from utils.setup import detect_setup
from utils.entry import calc_entry
from utils.rr_ev import compute_trade_plan
from utils.sector import top_sectors_5d, sector_rank_map
from utils.diversify import diversify_select

PRICE_MIN = 200
PRICE_MAX = 15000
ADV20_MIN = 200_000_000
ATR_PCT_MIN = 0.015
ATR_PCT_MAX = 0.060

RR_MIN = 2.2
EXPECTED_DAYS_MAX = 5.0
R_PER_DAY_MIN = 0.5

EARNINGS_EXCLUDE_DAYS = 3
SECTOR_TOP_N = 5
MAX_FINAL = 5
MAX_WATCH = 10

def _fetch_history(ticker: str, period: str = "260d") -> pd.DataFrame | None:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.4)
    return None

def _load_universe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _earnings_blocked(row: pd.Series, today_date) -> bool:
    if "earnings_date" not in row.index:
        return False
    d = pd.to_datetime(row.get("earnings_date", None), errors="coerce")
    if pd.isna(d):
        return False
    try:
        ed = d.date()
        return abs((ed - today_date).days) <= EARNINGS_EXCLUDE_DAYS
    except Exception:
        return False

def _regime_ev_min(mkt_score: int) -> float:
    if mkt_score < 50:
        return 0.5
    if mkt_score < 60:
        return 0.45
    return 0.4

def _no_trade_by_market(mkt_score: int, delta3d: int) -> Tuple[bool, str]:
    if mkt_score < 45:
        return True, "MarketScore<45"
    if delta3d <= -5 and mkt_score < 55:
        return True, "Δ3d<=-5 & MarketScore<55"
    return False, ""

def run_swing(
    universe_path: str,
    events_path: str,
    today_date,
    mkt_score: int,
    delta3d: int,
) -> Dict:
    sectors_top = top_sectors_5d(top_n=SECTOR_TOP_N, universe_path=universe_path)
    rank_map = sector_rank_map(sectors_top)
    allowed_sectors = set([s for s, _ in sectors_top])

    major_event_tomorrow = is_major_event_tomorrow(events_path, today_date)
    no_trade_mkt, reason_mkt = _no_trade_by_market(mkt_score, delta3d)

    uni = _load_universe(universe_path)
    if uni.empty:
        return dict(trade_ok=False, no_trade_reason="Universe読み込み失敗", sectors_top=sectors_top, candidates=[], watchlist=[])

    t_col = "ticker" if "ticker" in uni.columns else ("code" if "code" in uni.columns else None)
    if t_col is None:
        return dict(trade_ok=False, no_trade_reason="ticker列なし", sectors_top=sectors_top, candidates=[], watchlist=[])

    # index close for RS feature
    try:
        idx_df = yf.Ticker("^TOPX").history(period="260d", auto_adjust=True)
        idx_close = idx_df["Close"].astype(float) if idx_df is not None and not idx_df.empty else None
    except Exception:
        idx_close = None

    raw: List[Dict] = []
    histories: Dict[str, pd.DataFrame] = {}

    gu_count = 0
    gu_total = 0
    watch: List[Dict] = []

    ev_min = _regime_ev_min(mkt_score)

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = _fetch_history(ticker, period="260d")
        if hist is None or len(hist) < 120:
            continue

        f = compute_features(hist, index_close=idx_close)
        if f is None:
            continue

        if not (PRICE_MIN <= f.price <= PRICE_MAX):
            continue

        if not (np.isfinite(f.turnover_ma20) and f.turnover_ma20 >= ADV20_MIN):
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "流動性弱(ADV20<200M)"})
            continue

        if not (ATR_PCT_MIN <= f.atr_pct <= ATR_PCT_MAX):
            continue

        if _earnings_blocked(row, today_date):
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "決算近接(新規禁止)"})
            continue

        if allowed_sectors and sector not in allowed_sectors:
            continue

        s = detect_setup(hist, f)
        if s is None:
            continue

        e = calc_entry(hist, f, s)
        if e is None:
            continue

        gu_total += 1
        if e.gu_flag:
            gu_count += 1

        srank = int(rank_map.get(sector, 99))
        tp = compute_trade_plan(hist, f, s, e, mkt_score, delta3d, srank, major_event_tomorrow)
        if tp is None:
            continue

        if tp.rr < RR_MIN:
            continue
        if tp.ev < ev_min:
            continue
        if tp.expected_days > EXPECTED_DAYS_MAX:
            continue
        if tp.r_per_day < R_PER_DAY_MIN:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "速度/効率"})
            continue

        raw.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                setup=s.setup,
                sector_rank=srank,
                in_center=e.in_center,
                in_low=e.in_low,
                in_high=e.in_high,
                price_now=f.price,
                atr=f.atr,
                gu_flag=e.gu_flag,
                stop=tp.stop,
                tp1=tp.tp1,
                tp2=tp.tp2,
                rr=tp.rr,
                expected_days=tp.expected_days,
                r_per_day=tp.r_per_day,
                ev=tp.ev,
                adjev=tp.adjev,
                action=e.action,
            )
        )
        histories[ticker] = hist

    raw.sort(key=lambda x: (x["adjev"], x["r_per_day"], x["rr"]), reverse=True)

    a_list = [c for c in raw if c["setup"] == "A"]
    b_list = [c for c in raw if c["setup"] == "B"]
    balanced: List[Dict] = []
    if b_list:
        balanced.append(b_list[0])
    balanced.extend(a_list[:3])
    seen = set([c["ticker"] for c in balanced])
    for c in raw:
        if c["ticker"] in seen:
            continue
        balanced.append(c)
        if len(balanced) >= max(10, MAX_FINAL * 2):
            break

    selected, dropped = diversify_select(balanced, histories, max_final=MAX_FINAL)

    for d in dropped:
        watch.append({"ticker": d["ticker"], "name": d["name"], "sector": d["sector"], "reason": d.get("drop_reason", "制約")})

    wmap = {}
    for w in watch:
        if w["ticker"] not in wmap:
            wmap[w["ticker"]] = w
    watchlist = list(wmap.values())[:MAX_WATCH]

    gu_ratio = (gu_count / gu_total) if gu_total > 0 else 0.0
    no_trade = False
    no_trade_reason = ""
    if no_trade_mkt:
        no_trade = True
        no_trade_reason = reason_mkt
    else:
        best_adjev = float(selected[0]["adjev"]) if selected else -999.0
        if best_adjev < 0.6:
            no_trade = True
            no_trade_reason = "bestAdjEV<0.6"
        elif gu_total > 0 and gu_ratio >= 0.60:
            no_trade = True
            no_trade_reason = "GU比率>=60%"

    return dict(
        trade_ok=(not no_trade),
        no_trade_reason=no_trade_reason,
        sectors_top=sectors_top,
        candidates=selected,
        watchlist=watchlist,
    )
