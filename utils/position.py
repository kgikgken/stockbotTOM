from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from utils.scoring import calc_inout_for_stock


def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _fetch_hist(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period="130d", auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def analyze_positions(df: pd.DataFrame, mkt_score: int = 50):
    """
    return: (text, total_asset)
    """
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0

    lines = []
    total = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry = float(row.get("entry_price", 0) or 0)
        qty = float(row.get("quantity", 0) or 0)

        hist = _fetch_hist(ticker)
        if hist is None or len(hist) < 20:
            cur = float(row.get("current_price", entry) or entry)
            pnl_pct = (cur - entry) / entry * 100 if entry > 0 else 0.0
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")
            total += max(cur, 0.0) * max(qty, 0.0)
            continue

        cur = float(hist["Close"].astype(float).iloc[-1])
        pnl_pct = (cur - entry) / entry * 100 if entry > 0 else 0.0

        # ポジションRR更新（同じTP/SLロジックでRR算出）
        in_rank, tp_pct, sl_pct = calc_inout_for_stock(hist)
        rr = (tp_pct / 100.0) / abs(sl_pct / 100.0) if sl_pct < 0 else 0.0

        lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}% RR:{rr:.2f}R")
        total += max(cur, 0.0) * max(qty, 0.0)

    text = "\n".join(lines) if lines else "ノーポジション"
    if not np.isfinite(total) or total <= 0:
        total = 2_000_000.0

    return text, float(total)