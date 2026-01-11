from __future__ import annotations

from typing import Tuple, List
import pandas as pd
import numpy as np
import yfinance as yf

from utils.setup import detect_setup
from utils.rr_ev import compute_exit_levels, compute_ev_metrics
from utils.market import rr_min_for_market

def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def analyze_positions(df: pd.DataFrame, today_date, mkt: dict, macro_on: bool) -> Tuple[str, float]:
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0

    mkt_score = int(mkt.get("score", 50) or 50)
    rr_min = rr_min_for_market(mkt_score)

    lines: List[str] = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        qty = float(row.get("quantity", 0) or 0)

        cur = np.nan
        try:
            h = yf.Ticker(ticker).history(period="10d", auto_adjust=True)
            if h is not None and not h.empty:
                cur = float(h["Close"].iloc[-1])
        except Exception:
            pass

        if np.isfinite(cur) and qty > 0:
            value = cur * qty
            if np.isfinite(value) and value > 0:
                total_value += value

        adjev = np.nan
        rr = np.nan
        try:
            hist = yf.Ticker(ticker).history(period="260d", auto_adjust=True)
            if hist is not None and len(hist) >= 80:
                setup, anchors, gu = detect_setup(hist)
                exits = compute_exit_levels(hist, anchors.get("entry_mid", float(cur) if np.isfinite(cur) else 0.0), anchors.get("atr", float(cur) * 0.02 if np.isfinite(cur) else 1.0))
                rr = float(exits["rr"])
                _, adjev, _, _ = compute_ev_metrics(
                    setup, rr, anchors.get("atr", float(cur) * 0.02 if np.isfinite(cur) else 1.0), anchors.get("entry_mid", float(cur) if np.isfinite(cur) else 0.0), exits["tp2"], mkt_score, macro_on, gu
                )
        except Exception:
            pass

        note = ""
        if np.isfinite(adjev) and adjev < 0.50:
            note = "（要注意）"
        if np.isfinite(rr) and rr < rr_min:
            note = "（要注意）"

        if np.isfinite(rr) and np.isfinite(adjev):
            lines.append(f"- {ticker}: RR:{rr:.2f} AdjEV:{adjev:.2f}{note}")
        else:
            lines.append(f"- {ticker}{note}")

    asset = total_value if total_value > 0 else 2_000_000.0
    return "\n".join(lines) if lines else "ノーポジション", float(asset)
