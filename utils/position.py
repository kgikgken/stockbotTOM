from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .rr import compute_rr


def load_positions(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def _fetch_last_price(ticker: str) -> float:
    try:
        df = yf.Ticker(ticker).history(period="5d")
        if df is None or df.empty:
            return np.nan
        return float(df["Close"].iloc[-1])
    except Exception:
        return np.nan


def analyze_positions(df: pd.DataFrame, mkt_score: int | None = None) -> Tuple[str, float]:
    """
    戻り値:
      text: LINE表示用テキスト
      total_asset: 推定総資産
    """
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_val = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry = float(row.get("entry_price", 0.0) or 0.0)
        qty = float(row.get("quantity", 0.0) or 0.0)
        cur_price = row.get("current_price", np.nan)

        if not np.isfinite(cur_price) or cur_price <= 0:
            cur_price = _fetch_last_price(ticker)
        if not np.isfinite(cur_price) or cur_price <= 0:
            cur_price = entry

        pnl_pct = 0.0
        if entry > 0:
            pnl_pct = (cur_price - entry) / entry * 100.0

        val = cur_price * qty
        total_val += max(val, 0.0)

        # RR再計算（あれば）
        rr_str = ""
        if mkt_score is not None:
            try:
                hist = yf.Ticker(ticker).history(period="130d")
                if hist is not None and len(hist) >= 40:
                    rr_info = compute_rr(hist, int(mkt_score))
                    rr_val = float(rr_info["rr"])
                    if rr_val > 0:
                        rr_str = f" RR:{rr_val:.2f}R"
            except Exception:
                pass

        lines.append(
            f"- {ticker}: 損益 {pnl_pct:.2f}%{rr_str}"
        )

    if total_val <= 0:
        total_val = 2_000_000.0

    text = "\n".join(lines) if lines else "ノーポジション"
    return text, float(total_val)