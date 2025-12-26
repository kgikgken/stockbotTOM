from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple

from utils.rr import compute_trade_plan

def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def analyze_positions(
    df: pd.DataFrame,
    mkt_score: int = 50
) -> Tuple[str, float]:
    """
    戻り値:
      - 表示用テキスト
      - 推定総資産（最低ラインあり）
    """
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry = float(row.get("entry_price", 0) or 0)
        qty = float(row.get("quantity", 0) or 0)

        if entry <= 0 or qty <= 0:
            continue

        cur = entry
        try:
            h = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if h is not None and not h.empty:
                cur = float(h["Close"].iloc[-1])
        except Exception:
            pass

        pnl_pct = (cur - entry) / entry * 100.0
        value = cur * qty
        if np.isfinite(value):
            total_value += value

        # RR 再計算（毎朝アップデート）
        rr = 0.0
        try:
            hist = yf.Ticker(ticker).history(period="300d", auto_adjust=True)
            if hist is not None and len(hist) >= 120:
                plan = compute_trade_plan(hist, "A", mkt_score)
                rr = float(plan.get("R", 0.0))
        except Exception:
            pass

        if rr > 0:
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}% RR:{rr:.2f}R")
        else:
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")

    if not lines:
        return "ノーポジション", 2_000_000.0

    asset_est = max(total_value, 2_000_000.0)
    return "\n".join(lines), float(asset_est)