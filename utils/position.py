from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


# ============================================================
# CSV読み込み
# ============================================================
def load_positions(path: str) -> pd.DataFrame:
    """
    positions.csv を読み込む。
    なければ空 DataFrame を返す。
    必須列:
      ticker, avg_price, qty
    任意列:
      tp_pct, sl_pct
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ticker", "avg_price", "qty"])

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["ticker", "avg_price", "qty"])

    # 型安全化
    df["ticker"] = df.get("ticker", "").astype(str)
    df["avg_price"] = pd.to_numeric(df.get("avg_price", 0), errors="coerce")
    df["qty"] = pd.to_numeric(df.get("qty", 0), errors="coerce")

    # TP/SL（無い場合は None）
    df["tp_pct"] = pd.to_numeric(df.get("tp_pct", np.nan), errors="coerce")
    df["sl_pct"] = pd.to_numeric(df.get("sl_pct", np.nan), errors="coerce")

    # 無効行 drop
    df = df.dropna(subset=["avg_price", "qty"], how="any")
    return df.reset_index(drop=True)


# ============================================================
# ポジション分析
# ============================================================
def _fetch_price(ticker: str) -> float:
    """
    ticker の現在値を返す。
    yfinance を避けて main 側から hist の終値を受ける想定だったが、
    ここは簡易に yf を直読みしてもOK。
    将来 main で渡す場合は、この関数を削除。
    """
    import yfinance as yf
    try:
        hist = yf.Ticker(ticker).history(period="2d")
        if hist is None or len(hist) == 0:
            return np.nan
        c = float(hist["Close"].iloc[-1])
        return float(c)
    except Exception:
        return np.nan


def analyze_positions(df: pd.DataFrame) -> Tuple[str, float, float, float, Dict]:
    """
    ポジションを解析してまとめを返す。

    戻り値:
      pos_text:  テキスト（LINE貼り付け用）
      total_asset: 総資産（現金＋評価額）
      total_pos: 建玉総額
      lev: レバレッジ
      risk_info: dict（将来用）
    """
    if df is None or len(df) == 0:
        # ノーポジション
        text = "ノーポジション"
        return text, 2_000_000.0, 0.0, 1.0, {"pos": []}

    lines: List[str] = []
    total_pos = 0.0
    total_pl = 0.0
    detail = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        avg = float(row["avg_price"])
        qty = float(row["qty"])
        tp_pct = row.get("tp_pct", np.nan)
        sl_pct = row.get("sl_pct", np.nan)

        now = _fetch_price(ticker)
        if not np.isfinite(now):
            continue

        value = now * qty
        pos_pl = (now - avg) * qty
        pl_pct = (now / avg - 1.0) * 100.0

        total_pos += value
        total_pl += pos_pl

        # テキスト
        line = f"- {ticker}: {now:.1f} / {avg:.1f} / 損益 {pl_pct:+.2f}%"
        if np.isfinite(tp_pct) and np.isfinite(sl_pct):
            tp_price = avg * (1.0 + tp_pct)
            sl_price = avg * (1.0 + sl_pct)
            line += f"\n    TP:{tp_price:.1f} SL:{sl_price:.1f}"
        lines.append(line)

        detail.append({
            "ticker": ticker,
            "avg": avg,
            "qty": qty,
            "now": now,
            "pl_pct": pl_pct,
            "value": value,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
        })

    # 総資産 = assumed cash + 評価損益
    est_cash = 2_000_000.0
    total_asset = est_cash + total_pl

    # レバレッジ
    if total_asset > 0:
        lev = total_pos / total_asset
    else:
        lev = 1.0

    text = "\n".join(lines)
    return text, float(total_asset), float(total_pos), float(lev), {"pos": detail}