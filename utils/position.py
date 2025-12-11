from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .scoring import calc_inout_for_stock


def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _fetch_hist(ticker: str, period: str = "90d") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None


def _calc_rr_for_position(ticker: str, entry_price: float) -> float:
    """
    現在のボラ・押し目条件からざっくりRRを推定
    （新規候補と同じロジック）
    """
    if entry_price <= 0:
        return np.nan

    hist = _fetch_hist(ticker)
    if hist is None or len(hist) < 60:
        return np.nan

    _, tp_pct, sl_pct = calc_inout_for_stock(hist)
    if sl_pct >= 0:
        return np.nan

    rr = (tp_pct / 100.0) / abs(sl_pct / 100.0)
    return float(rr)


def analyze_positions(df: pd.DataFrame, mkt_score: int = 50) -> Tuple[str, float]:
    """
    positions.csv を解析して
    - LINE用のポジション文字列
    - 総資産推定
    を返す
    """
    if df is None or len(df) == 0:
        text = "ノーポジション"
        asset = 2_000_000.0
        return text, float(asset)

    lines = []
    total = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", row.get("code", "")))
        entry = float(row.get("entry_price", 0) or 0)
        qty = float(row.get("quantity", row.get("qty", 0)) or 0)
        price = float(row.get("current_price", entry) or entry)

        pnl_pct = (price - entry) / entry * 100.0 if entry > 0 else 0.0
        value = price * qty
        total += value

        rr_now = _calc_rr_for_position(ticker, entry)
        if np.isfinite(rr_now):
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}% RR:{rr_now:.2f}R")
        else:
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")

    if total <= 0:
        total = 2_000_000.0

    text = "\n".join(lines) if lines else "ノーポジション"

    return text, float(total)