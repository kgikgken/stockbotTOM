from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.rr import compute_rr

DEFAULT_ASSET = 3_000_000  # ポジがないときの仮の運用資産


def load_positions(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame(columns=["ticker", "shares", "entry_price"])


def analyze_positions(df: pd.DataFrame) -> Tuple[str, float]:
    """
    positions.csv を解析して
      - LINE用ポジション文字列
      - 推定運用資産
    を返す
    """
    if df is None or df.empty:
        return "ノーポジション", float(DEFAULT_ASSET)

    lines = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        shares = float(row.get("shares", 0))
        entry_price = float(row.get("entry_price", 0))

        if not ticker or shares <= 0 or entry_price <= 0:
            continue

        # 現在値取得
        try:
            hist = yf.download(
                ticker,
                period="60d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if hist is None or hist.empty:
                last_close = entry_price
            else:
                last_close = float(hist["Close"].astype(float).iloc[-1])
        except Exception:
            last_close = entry_price
            hist = None

        pnl_pct = (last_close - entry_price) / entry_price * 100.0

        # 既存ポジションの RR（entry_price 固定）
        try:
            if hist is not None:
                rr_info = compute_rr(hist, mkt_score=50, entry_price=entry_price)
                rr_val = float(rr_info.get("rr", 0.0))
            else:
                rr_val = 0.0
        except Exception:
            rr_val = 0.0

        value = shares * last_close
        total_value += value

        lines.append(f"- {ticker}: 損益 {pnl_pct:.2f}% RR:{rr_val:.2f}R")

    if not lines:
        return "ノーポジション", float(DEFAULT_ASSET)

    asset = total_value if total_value > 0 else DEFAULT_ASSET
    text = "\n".join(lines)
    return text, float(asset)