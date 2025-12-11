from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _latest_price(ticker: str) -> float:
    try:
        df = yf.Ticker(ticker).history(period="5d")
        if df is None or df.empty:
            return np.nan
        return float(df["Close"].astype(float).iloc[-1])
    except Exception:
        return np.nan


def analyze_positions(df: pd.DataFrame, mkt_score: int = 50) -> Tuple[str, float]:
    """
    positions.csv を解析して
    - LINE用テキスト
    - 総資産推定
    を返す
    """
    if df is None or df.empty:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", row.get("code", "")))
        if not ticker:
            continue

        shares = float(row.get("shares", row.get("quantity", 0.0)) or 0.0)
        entry = float(row.get("entry_price", row.get("entry", 0.0)) or 0.0)

        if shares <= 0 or entry <= 0:
            lines.append(f"- {ticker}: 損益 0.00% RR:2.00R")
            continue

        cur_price = row.get("current_price", np.nan)
        try:
            cur_price = float(cur_price)
        except Exception:
            cur_price = np.nan

        if not np.isfinite(cur_price) or cur_price <= 0:
            cur_price = _latest_price(ticker)

        if not np.isfinite(cur_price) or cur_price <= 0:
            cur_price = entry

        pnl_pct = (cur_price - entry) / entry * 100.0
        value = cur_price * shares
        total_value += value

        rr_now = 2.0

        lines.append(
            f"- {ticker}: 損益 {pnl_pct:.2f}% RR:{rr_now:.2f}R"
        )

    if total_value <= 0:
        total_value = 2_000_000.0

    text = "\n".join(lines) if lines else "ノーポジション"
    return text, float(total_value)