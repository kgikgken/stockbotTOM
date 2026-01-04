from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple
from datetime import timedelta

import yfinance as yf

from utils.util import safe_float


def load_positions(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def analyze_positions(df: pd.DataFrame, mkt_score: int = 50) -> Tuple[str, float]:
    if df is None or len(df) == 0:
        return "ノーポジション", 2_000_000.0

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        entry = safe_float(row.get("entry_price", 0), 0.0)
        qty = safe_float(row.get("quantity", 0), 0.0)

        cur = entry
        try:
            h = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if h is not None and not h.empty:
                cur = float(h["Close"].iloc[-1])
        except Exception:
            pass

        pnl_pct = (cur - entry) / entry * 100.0 if entry > 0 else np.nan
        value = cur * qty
        if np.isfinite(value) and value > 0:
            total_value += value

        if np.isfinite(pnl_pct):
            lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}%")
        else:
            lines.append(f"- {ticker}: 損益 n/a")

    if not lines:
        return "ノーポジション", 2_000_000.0

    asset_est = total_value if total_value > 0 else 2_000_000.0
    return "\n".join(lines), float(asset_est)


def weekly_new_count(df: pd.DataFrame, today_date) -> int:
    """
    positions.csv に open_date (YYYY-MM-DD) がある想定。
    無い場合は 0 扱い（安全側）
    """
    if df is None or len(df) == 0:
        return 0
    if "open_date" not in df.columns:
        return 0

    try:
        d = pd.to_datetime(df["open_date"], errors="coerce").dt.date
    except Exception:
        return 0

    # 直近7日（暦週に厳密でなく、運用上の“週次制限”として安全側）
    start = today_date - timedelta(days=7)
    c = 0
    for x in d:
        if x is None or pd.isna(x):
            continue
        if start <= x <= today_date:
            c += 1
    return int(c)


def lot_accident_warning(picked, total_asset: float, risk_per_trade: float = 0.015) -> str:
    """
    “最大同時損失” を雑に見積もって事故を警告
    - 1銘柄の想定損失 = 資産 * risk_per_trade
    - 同時に最大5銘柄が逆行したら？を表示
    """
    if not picked:
        return ""

    n = len(picked)
    max_loss = float(total_asset * risk_per_trade * n)
    pct = (max_loss / total_asset * 100.0) if total_asset > 0 else 0.0

    # 8%超で強警戒
    if pct >= 8.0:
        return f"⚠ ロット事故警告：想定最大損失 ≈ {int(max_loss):,}円（資産比 {pct:.2f}%）"
    return ""