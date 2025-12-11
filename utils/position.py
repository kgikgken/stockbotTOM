from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .scoring import calc_inout_for_stock


def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def _current_price_for_ticker(ticker: str) -> float:
    try:
        df = yf.Ticker(ticker).history(period="3d")
        if df is None or df.empty:
            return np.nan
        close = df["Close"].astype(float)
        return float(close.iloc[-1])
    except Exception:
        return np.nan


def analyze_positions(df: pd.DataFrame, mkt_score: int = 50) -> Tuple[str, float]:
    if df is None or df.empty:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry = float(row.get("entry_price", 0.0))
        qty = float(row.get("quantity", 0.0))
        cur_price = row.get("current_price", np.nan)
        try:
            cur_price = float(cur_price)
        except Exception:
            cur_price = np.nan

        if not np.isfinite(cur_price) or cur_price <= 0:
            cur_price = _current_price_for_ticker(ticker)
        if not np.isfinite(cur_price) or cur_price <= 0:
            cur_price = entry

        pnl_pct = (cur_price - entry) / entry * 100.0 if entry > 0 else 0.0

        # RR 推定（現状のチャートから再計算）
        rr_txt = ""
        try:
            hist = yf.Ticker(ticker).history(period="130d")
            if hist is not None and not hist.empty and entry > 0:
                in_rank, tp_pct, sl_pct = calc_inout_for_stock(hist)
                rr = (tp_pct / 100.0) / abs(sl_pct / 100.0) if sl_pct < 0 else 0.0
                rr_txt = f" RR:{rr:.2f}R"
        except Exception:
            rr_txt = ""

        value = qty * cur_price
        if np.isfinite(value) and value > 0:
            total_value += value

        lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%{rr_txt}")

    if total_value <= 0:
        total_value = 2_000_000.0

    text = "\n".join(lines) if lines else "ノーポジション"
    return text, float(total_value)