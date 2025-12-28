from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
import yfinance as yf

from utils.util import read_csv_safely, safe_float
from utils.rr_ev import compute_rr_targets
from utils.entry import calc_entry_zone
from utils.setup import detect_setup


POSITIONS_PATH = "positions.csv"


def load_positions(path: str = POSITIONS_PATH) -> pd.DataFrame:
    return read_csv_safely(path)


def analyze_positions(df: pd.DataFrame, mkt_score: int = 50) -> Tuple[str, float]:
    """
    positions.csv: ticker, entry_price, quantity（無くても落ちない）
    戻り: (pos_text, total_asset_est)
    """
    if df is None or df.empty:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry_price = safe_float(row.get("entry_price", 0), 0.0)
        qty = safe_float(row.get("quantity", 0), 0.0)

        cur = entry_price
        try:
            h = yf.Ticker(ticker).history(period="6d", auto_adjust=True)
            if h is not None and not h.empty:
                cur = float(h["Close"].iloc[-1])
        except Exception:
            pass

        pnl_pct = (cur - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0
        value = cur * qty
        if np.isfinite(value) and value > 0:
            total_value += value

        # RR更新（保有分も毎日）
        rr = 0.0
        try:
            hist = yf.Ticker(ticker).history(period="260d", auto_adjust=True)
            if hist is not None and len(hist) >= 80:
                setup = detect_setup(hist).get("setup", "-")
                entry = calc_entry_zone(hist, setup)
                rr_info = compute_rr_targets(hist, setup, entry)
                rr = float(rr_info.get("rr", 0.0))
        except Exception:
            rr = 0.0

        if rr > 0:
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}% RR:{rr:.2f}R")
        else:
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")

    if not lines:
        return "ノーポジション", 2_000_000.0

    asset_est = total_value if total_value > 0 else 2_000_000.0
    return "\n".join(lines), float(asset_est)