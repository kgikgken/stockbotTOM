from __future__ import annotations

import os
from typing import Tuple, Dict, List
from datetime import datetime

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

    # 必須列チェック
    for col in ["ticker", "size", "entry_price"]:
        if col not in df.columns:
            return pd.DataFrame(columns=["ticker", "size", "entry_price"])

    return df


# ============================================================
# 現値取得
# ============================================================
def fetch_price(ticker: str) -> float:
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return np.nan
        return float(df["Close"].iloc[-1])
    except Exception:
        return np.nan


# ============================================================
# 60日ヒストリー（RR計算用）
# ============================================================
def fetch_hist(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


# ============================================================
# RR計算
# ============================================================
def analyze_positions(
    df: pd.DataFrame,
    mkt_score: int = 50
) -> Tuple[str, float, float, float, List[Dict]]:
    """
    positions.csv を解析し、
    - ポジション要約テキスト
    - 推定資産（評価額 + 現金）
    - 総ポジション額
    - 推定レバレッジ
    - 詳細RR情報
    を返す。
    """

    # 例外：ノーポジ
    if df is None or len(df) == 0:
        text = "ノーポジション"
        return text, 2_000_000.0, 0.0, 0.0, []

    details: List[Dict] = []
    total_value = 0.0

    # 現在価格ベース計算
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        size = float(row["size"])
        entry = float(row["entry_price"])

        price = fetch_price(ticker)
        if not np.isfinite(price):
            continue

        # 評価額
        pos_val = price * size
        total_value += pos_val

        # TP/SL/RR算出
        hist = fetch_hist(ticker)
        tp_pct, sl_pct, rr = compute_tp_sl_rr(hist, mkt_score)

        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)

        # RR（現値ベース）
        if price > 0 and sl_price < price:
            rr_now = (tp_price - price) / (price - sl_price)
        else:
            rr_now = rr  # fallback

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

    # 推定資産：現金 + 評価額
    # → 現金は仮定（今は必要最小限のモデル）
    cash = 500_000.0  # 後でこれを正確に入れる
    total_asset = total_value + cash

    # 総ポジション
    total_pos = total_value

    # レバレッジ: 総ポジション / 現金
    lev = total_pos / cash if cash > 0 else 1.0

    # LINE用テキスト
    lines = []
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