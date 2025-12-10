from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.rr import compute_rr


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
            return float("nan")
        return float(df["Close"].iloc[-1])
    except Exception:
        return float("nan")


def analyze_positions_with_rr(
    df: pd.DataFrame,
    mkt_score: int,
) -> Tuple[str, float, Dict[str, float]]:
    """
    positions.csv を解析して
      - LINE表示用テキスト
      - 推定総資産
      - {ticker: rr} のmap
    を返す
    """
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0, {}

    lines = []
    total = 0.0
    rr_map: Dict[str, float] = {}

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry = float(row.get("entry_price", 0.0) or 0.0)
        qty = float(row.get("quantity", 0.0) or 0.0)

        cur = row.get("current_price", np.nan)
        try:
            cur_price = float(cur)
        except Exception:
            cur_price = np.nan

        if not np.isfinite(cur_price) or cur_price <= 0:
            cur_price = _fetch_last_price(ticker)

        if not np.isfinite(cur_price) or cur_price <= 0:
            cur_price = entry

        value = qty * cur_price
        total += value

        pnl_pct = (cur_price - entry) / entry * 100.0 if entry > 0 else 0.0

        # RR再計算
        rr = np.nan
        try:
            hist = yf.Ticker(ticker).history(period="80d")
            if hist is not None and len(hist) >= 40:
                rr_info = compute_rr(hist, mkt_score, in_rank=None)
                rr = float(rr_info["rr"])
        except Exception:
            rr = np.nan

        if np.isfinite(rr):
            rr_map[ticker] = rr
            rr_part = f" RR:{rr:.2f}R"
        else:
            rr_part = ""

        lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%{rr_part}")

    text = "\n".join(lines) if lines else "ノーポジション"

    if total <= 0:
        total = 2_000_000.0

    return text, float(total), rr_map