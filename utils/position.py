from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.setup import classify_setup
from utils.rr_ev import compute_trade_plan


def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def analyze_positions(df: pd.DataFrame, mkt_score: int, macro_caution: bool) -> Tuple[str, float]:
    """Return (text, asset_est).

    positions.csv expected columns: ticker, entry_price, quantity (optional).
    If missing, still works.
    """
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry_price = float(row.get("entry_price", 0) or 0)
        qty = float(row.get("quantity", 0) or 0)

        cur = entry_price
        try:
            h = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if h is not None and not h.empty:
                cur = float(h["Close"].iloc[-1])
        except Exception:
            pass

        value = cur * qty
        if np.isfinite(value) and value > 0:
            total_value += value

        rr = float("nan")
        adjev = float("nan")
        try:
            hist = yf.Ticker(ticker).history(period="260d", auto_adjust=True)
            if hist is not None and len(hist) >= 80:
                s = classify_setup(hist)
                plan = compute_trade_plan(
                    df=hist,
                    setup=s.name,
                    atr=s.atr,
                    sma20=s.sma20,
                    mkt_score=mkt_score,
                    macro_caution=macro_caution,
                    allow_tp2_tight=macro_caution,
                )
                rr = float(plan.get("rr", float("nan")))
                adjev = float(plan.get("adjev", float("nan")))
        except Exception:
            pass

        if np.isfinite(rr) and np.isfinite(adjev):
            lines.append(f"- {ticker}: RR:{rr:.2f} 期待値:{adjev:+.2f}（注意）" if adjev < 0.50 else f"- {ticker}: RR:{rr:.2f} 期待値:{adjev:+.2f}")
        elif np.isfinite(rr):
            lines.append(f"- {ticker}: RR:{rr:.2f}")
        else:
            lines.append(f"- {ticker}")

    asset_est = float(total_value) if total_value > 0 else 2_000_000.0
    return ("\n".join(lines) if lines else "ノーポジション"), asset_est
