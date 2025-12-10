from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.scoring import calc_inout_for_stock


def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def analyze_positions(df: pd.DataFrame, mkt_score: int) -> Tuple[str, float]:
    """
    positions.csv を解析して
    - LINE用のポジション文字列
    - 総資産推定
    を返す

    必ず (text, asset) の 2値で return する
    """
    if df is None or len(df) == 0:
        text = "ノーポジション"
        asset = 2_000_000
        return text, float(asset)

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry = float(row.get("entry_price", 0) or 0)
        qty = float(row.get("quantity", 0) or 0)

        current_price = entry
        rr_val = np.nan
        in_rank = "N/A"

        try:
            hist = yf.Ticker(ticker).history(period="60d")
            if hist is not None and not hist.empty:
                current_price = float(hist["Close"].astype(float).iloc[-1])
                in_rank, tp_pct, sl_pct = calc_inout_for_stock(hist)
                if sl_pct < 0:
                    rr_val = float((tp_pct / 100.0) / abs(sl_pct / 100.0))
        except Exception:
            pass

        if entry > 0 and np.isfinite(current_price):
            pnl_pct = (current_price - entry) / entry * 100.0
        else:
            pnl_pct = 0.0

        value = qty * current_price
        total_value += value

        if np.isfinite(rr_val):
            lines.append(
                f"- {ticker}: 損益 {pnl_pct:.2f}% RR:{rr_val:.2f}R ({in_rank})"
            )
        else:
            lines.append(
                f"- {ticker}: 損益 {pnl_pct:.2f}%"
            )

    text = "\n".join(lines) if lines else "ノーポジション"

    if total_value <= 0:
        total_value = 2_000_000.0

    return text, float(total_value)