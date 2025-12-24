from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.rr import classify_setup, build_trade_plan


def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def analyze_positions(df: pd.DataFrame, mkt_score: int = 50) -> Tuple[str, float]:
    """
    positions.csv は最低限 ticker, entry_price, quantity を想定（無くても落ちない）
    戻り値: (pos_text, total_asset_est)
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
            h = yf.Ticker(ticker).history(period="8d", auto_adjust=True)
            if h is not None and not h.empty:
                cur = float(h["Close"].iloc[-1])
        except Exception:
            pass

        pnl_pct = (cur - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0
        value = cur * qty
        if np.isfinite(value) and value > 0:
            total_value += value

        rr_txt = ""
        try:
            hist = yf.Ticker(ticker).history(period="320d", auto_adjust=True)
            if hist is not None and len(hist) >= 120:
                setup = classify_setup(hist)
                if setup:
                    plan = build_trade_plan(hist, setup=setup)
                    if plan and plan.r > 0:
                        rr_txt = f" R:{plan.r:.2f}"
        except Exception:
            rr_txt = ""

        lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%{rr_txt}")

    if not lines:
        return "ノーポジション", 2_000_000.0

    asset_est = total_value if total_value > 0 else 2_000_000.0
    return "\n".join(lines), float(asset_est)
