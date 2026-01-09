from __future__ import annotations

from datetime import date
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from utils.setup import detect_setup
from utils.rr_ev import build_trade_plan


def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def analyze_positions_summary(df: pd.DataFrame, today_date: date, mkt_score: int = 50) -> Tuple[str, float, Optional[int]]:
    """
    Returns:
      - positions text
      - total_asset_est (rough)
      - weekly_new_count (if entry_date column exists; else None)

    positions.csv optional columns:
      ticker, entry_price, quantity, entry_date (YYYY-MM-DD)

    仕様の「週次新規回数：最大3」用の表示値。
    （厳密な新規回数管理は永続状態が必要なので、entry_date列がある場合のみ集計）
    """
    if df is None or df.empty:
        return "ノーポジション", 2_000_000.0, None

    lines = []
    total_value = 0.0

    weekly_new = None
    if "entry_date" in df.columns:
        try:
            ed = pd.to_datetime(df["entry_date"], errors="coerce")
            iso_week = today_date.isocalendar().week
            iso_year = today_date.isocalendar().year
            weekly_new = int(((ed.dt.isocalendar().week == iso_week) & (ed.dt.isocalendar().year == iso_year)).sum())
        except Exception:
            weekly_new = None

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry_price = _safe_float(row.get("entry_price", 0) or 0, 0.0)
        qty = _safe_float(row.get("quantity", 0) or 0, 0.0)

        cur = entry_price
        try:
            h = yf.Ticker(ticker).history(period="10d", auto_adjust=True)
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
            hist = yf.Ticker(ticker).history(period="260d", auto_adjust=True)
            if hist is not None and len(hist) >= 80:
                s = detect_setup(hist)
                if s.setup != "NONE":
                    plan = build_trade_plan(
                        hist, s.setup, s.entry_low, s.entry_high, s.entry_mid, s.stop_seed, mkt_score=mkt_score, macro_on=False
                    )
                    if plan is not None:
                        rr_txt = f" RR:{plan.rr:.2f} AdjEV:{plan.adjev:.2f}"
        except Exception:
            rr_txt = ""

        lines.append(f"- {ticker}: 損益 {pnl_pct:+.2f}%{rr_txt}")

    asset_est = float(total_value) if total_value > 0 else 2_000_000.0
    if not lines:
        return "ノーポジション", asset_est, weekly_new

    return "\n".join(lines), asset_est, weekly_new
