from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional

import pandas as pd
import numpy as np
import yfinance as yf

from utils.setup import classify_setup, atr
from utils.rr_ev import compute_levels, rr as calc_rr, expected_days, speed_r_per_day, structural_ev, adj_ev_from_struct


@dataclass
class Candidate:
    ticker: str
    name: str
    sector: str
    setup: str
    entry_center: float
    entry_low: float
    entry_high: float
    rr: float
    adj_ev: float
    speed: float
    exp_days: float
    sl: float
    tp1: float
    tp2: float
    gu: bool


def _download_ohlc(ticker: str, today: date) -> Optional[pd.DataFrame]:
    start = pd.Timestamp(today) - pd.Timedelta(days=420)
    end = pd.Timestamp(today) + pd.Timedelta(days=1)
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        df = df.dropna()
        # 統一
        df = df.rename(columns={"Adj Close": "AdjClose"})
        return df
    except Exception:
        return None


def build_raw_candidates(today_date: date, universe: pd.DataFrame, rr_min: float, adj_ev_min: float, rday_min_by_setup: dict) -> List[Candidate]:
    out: List[Candidate] = []

    for _, row in universe.iterrows():
        ticker = str(row.get("ticker") or row.get("code") or "").strip()
        if not ticker:
            continue
        if not ticker.endswith(".T"):
            # JPXフォーマットに寄せる
            if ticker.isdigit():
                ticker = f"{ticker}.T"

        name = str(row.get("name") or row.get("company") or "").strip() or ticker
        sector = str(row.get("sector") or row.get("industry") or "").strip() or "-"

        df = _download_ohlc(ticker, today_date)
        if df is None or len(df) < 80:
            continue

        close = df["Close"]
        last = float(close.iloc[-1])
        if last < 200 or last > 15000:
            continue

        # ATR / vol filter
        atr14 = atr(df, 14)
        atr_v = float(atr14.iloc[-1]) if len(atr14.dropna()) else 0.0
        if atr_v <= 0:
            continue
        atr_pct = atr_v / last * 100
        if atr_pct < 1.5:
            continue

        # Liquidity (ADV20) - use Volume * Close (rough)
        try:
            adv20 = float((df["Volume"].tail(20) * df["Close"].tail(20)).mean())
        except Exception:
            adv20 = 0.0
        if adv20 < 200_000_000:
            continue

        # Setup
        s = classify_setup(df)
        if s.setup == "NONE":
            continue

        # Entry band internal (ATR cap by setup)
        sma20 = close.rolling(20).mean()
        entry_center = float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else last

        if s.setup == "A1-Strong":
            band = 0.35 * atr_v
            cap = 0.40 * atr_v
        elif s.setup == "A1":
            band = 0.40 * atr_v
            cap = 0.50 * atr_v
        else:
            band = 0.30 * atr_v
            cap = 0.30 * atr_v
        band = min(band, cap)
        entry_low = entry_center - band
        entry_high = entry_center + band

        # RR target base
        rr_target = max(rr_min, 2.2)
        levels = compute_levels(entry=entry_center, atr=atr_v, rr_target=rr_target)
        rr_value = calc_rr(levels)

        exp_d = expected_days(entry_center, levels.tp2, atr_v)
        spd = speed_r_per_day(rr_value, exp_d)

        rday_min = float(rday_min_by_setup.get(s.setup, 0.50))

        struct = structural_ev(rr_value, s.trend_strength, s.pullback_quality)
        aev = adj_ev_from_struct(struct)

        # GU判定（前日終値→当日始値）
        gu = False
        try:
            if len(df) >= 2:
                prev_close = float(df["Close"].iloc[-2])
                today_open = float(df["Open"].iloc[-1])
                gu = (today_open / prev_close - 1) >= 0.02
        except Exception:
            gu = False

        if rr_value < rr_min:
            continue
        if aev < adj_ev_min:
            continue
        if spd < rday_min:
            continue

        out.append(
            Candidate(
                ticker=ticker,
                name=name,
                sector=sector,
                setup=s.setup,
                entry_center=entry_center,
                entry_low=entry_low,
                entry_high=entry_high,
                rr=rr_value,
                adj_ev=aev,
                speed=spd,
                exp_days=exp_d,
                sl=levels.sl,
                tp1=levels.tp1,
                tp2=levels.tp2,
                gu=gu,
            )
        )

    # rank: adjEV then speed
    out.sort(key=lambda c: (c.adj_ev, c.speed, c.rr), reverse=True)
    return out
