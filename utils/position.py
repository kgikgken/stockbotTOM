from __future__ import annotations

import os
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from utils.rr import compute_tp_sl_rr


# ============================================================
# CSV 読み込み
# ============================================================
def load_positions(path: str) -> pd.DataFrame:
    """
    positions.csv 読み込み
    必須列:
      ticker, size, entry_price
    任意列:
      note
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ticker", "size", "entry_price"])

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["ticker", "size", "entry_price"])

    for col in ["ticker", "size", "entry_price"]:
        if col not in df.columns:
            return pd.DataFrame(columns=["ticker", "size", "entry_price"])

    return df


# ============================================================
# 現値・ヒストリー
# ============================================================
def fetch_price(ticker: str) -> float:
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return np.nan
        return float(df["Close"].iloc[-1])
    except Exception:
        return np.nan


def fetch_hist(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


# ============================================================
# ポジション解析（RR付き）
# ============================================================
def analyze_positions(
    df: pd.DataFrame,
    mkt_score: int = 50,
) -> Tuple[str, float, float, float, List[Dict]]:
    """
    positions.csv を解析し、
    - ポジション要約テキスト（LINE表示用）
    - 推定総資産
    - 総ポジション額
    - レバレッジ（総ポジション / 現金仮定）
    - 詳細RR情報リスト
    を返す。
    """

    # ノーポジ
    if df is None or len(df) == 0:
        text = "ノーポジション"
        return text, 2_000_000.0, 0.0, 0.0, []

    details: List[Dict] = []
    total_value = 0.0

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        try:
            size = float(row["size"])
            entry = float(row["entry_price"])
        except Exception:
            continue

        if not ticker or size == 0:
            continue

        price = fetch_price(ticker)
        if not np.isfinite(price) or price <= 0:
            continue

        pos_val = price * size
        total_value += pos_val

        hist = fetch_hist(ticker)
        tp_pct, sl_pct, rr_base = compute_tp_sl_rr(hist, mkt_score)

        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)

        # 現値ベースRR（今ここからのRR）
        if price > 0 and sl_price < price:
            rr_now = (tp_price - price) / (price - sl_price)
        else:
            rr_now = rr_base

        details.append(
            {
                "ticker": ticker,
                "size": size,
                "entry": entry,
                "price": price,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "rr": float(rr_now),
            }
        )

    # 現金は仮置き（後で正確に管理する前提）
    cash = 500_000.0
    total_asset = total_value + cash
    total_pos = total_value
    lev = total_pos / cash if cash > 0 else 1.0

    # LINE用テキスト
    lines: List[str] = []
    for d in details:
        lines.append(
            f"- {d['ticker']}: 現値 {d['price']:.1f} / IN {d['entry']:.1f}"
        )
        lines.append(
            f"    RR:{d['rr']:.2f}R  TP:{d['tp_price']:.1f}  SL:{d['sl_price']:.1f}"
        )
        lines.append("")

    text = "\n".join(lines).strip()
    return text, total_asset, total_pos, lev, details