from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional

import pandas as pd
import yfinance as yf

from utils.events import load_macro_events, macro_alert_on, MacroEvent
from utils.util import fmt_pct


@dataclass
class MarketSnapshot:
    market_score: float
    delta_score_3d: float
    futures_pct: float
    futures_symbol: str
    risk_text: str
    macro_on: bool
    events: List[MacroEvent]


def _calc_market_score(topix: pd.Series, nikkei: pd.Series) -> float:
    # シンプルに：MA構造 + 5日モメンタム
    score = 50.0
    try:
        def part(s: pd.Series) -> float:
            sma20 = s.rolling(20).mean()
            sma50 = s.rolling(50).mean()
            last = s.iloc[-1]
            p = 0.0
            if last > sma20.iloc[-1] > sma50.iloc[-1]:
                p += 15
            elif last > sma20.iloc[-1]:
                p += 7
            else:
                p -= 7
            mom = (s.iloc[-1] / s.iloc[-6] - 1) * 100 if len(s) >= 6 else 0.0
            p += max(min(mom * 3, 15), -15)
            return p

        score += 0.5 * part(topix) + 0.5 * part(nikkei)
    except Exception:
        pass

    return max(0.0, min(100.0, score))


def build_market_snapshot(today_date: date, events_path: str) -> MarketSnapshot:
    # Index data
    start = pd.Timestamp(today_date) - pd.Timedelta(days=120)
    end = pd.Timestamp(today_date) + pd.Timedelta(days=1)

    def fetch(sym: str) -> pd.Series:
        df = yf.download(sym, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df["Close"].dropna()

    topix = fetch("^TOPX")
    nikkei = fetch("^N225")

    score = _calc_market_score(topix, nikkei) if len(topix) and len(nikkei) else 50.0

    # delta 3d (score difference based on trailing window)
    delta3 = 0.0
    try:
        if len(topix) >= 4 and len(nikkei) >= 4:
            score_prev = _calc_market_score(topix.iloc[:-3], nikkei.iloc[:-3])
            delta3 = float(score - score_prev)
    except Exception:
        delta3 = 0.0

    # Futures risk-on/off
    fut_sym = "NKD=F"
    fut_pct = 0.0
    try:
        fut = yf.download(fut_sym, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        fut = fut["Close"].dropna()
        if len(fut) >= 2:
            fut_pct = float((fut.iloc[-1] / fut.iloc[-2] - 1) * 100)
    except Exception:
        fut_pct = 0.0

    risk_text = "Risk-ON" if fut_pct >= 1.0 else ("Risk-OFF" if fut_pct <= -1.0 else "中立")

    events = load_macro_events(events_path=events_path, today=today_date, lookahead_days=3)
    macro_on = macro_alert_on(events)

    return MarketSnapshot(
        market_score=float(round(score, 2)),
        delta_score_3d=float(round(delta3, 2)),
        futures_pct=float(round(fut_pct, 2)),
        futures_symbol=fut_sym,
        risk_text=risk_text,
        macro_on=macro_on,
        events=events,
    )
