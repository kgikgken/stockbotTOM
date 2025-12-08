# utils/position.py
# ============================================================
# ポジション管理 & 資産評価
# ============================================================

from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# 価格取得（安全版）
# ============================================================
def _fetch_price(ticker: str) -> float:
    """
    現値を取得（5日履歴）で終値を返す。
    板落ちやAPI死んでもNoneを返さずNaNで安全処理。
    """
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return np.nan
        price = df["Close"].iloc[-1]
        try:
            return float(price)
        except Exception:
            return np.nan
    except Exception:
        return np.nan


# ============================================================
# positions.csv 読み込み
# ============================================================
def load_positions(path: str) -> pd.DataFrame:
    """
    positions.csv を読み込んで返す。
    空ファイルでもOK（ノーポジ扱い）。
    """
    try:
        df = pd.read_csv(path)
        if df is None or df.empty:
            return pd.DataFrame(columns=["ticker", "shares", "price"])
        # 型を安全に
        df["ticker"] = df["ticker"].astype(str)
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).astype(float)
        return df
    except Exception:
        return pd.DataFrame(columns=["ticker", "shares", "price"])


# ============================================================
# ポジション分析
# ============================================================
def analyze_positions(df: pd.DataFrame):
    """
    ポジションを解析して:
        ① 報告文（text）
        ② 総資産（評価額ベース）
        ③ 総建玉
        ④ レバ（推定）
        ⑤ リスク情報
    を返す。

    戻り値:
        (text, total_asset, total_pos, lev, risk_info)
    """

    if df is None or df.empty:
        return "ノーポジション", 2_000_000.0, 0.0, 1.0, {}

    lines = []
    total_pos = 0.0
    total_asset = 0.0

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        shares = int(row["shares"])
        price_in = float(row["price"])

        price_now = _fetch_price(ticker)
        if not np.isfinite(price_now):
            continue

        pos_val = price_now * shares
        pnl = (price_now - price_in) / price_in if price_in > 0 else 0.0
        pnl_pct = pnl * 100.0

        total_pos += pos_val
        total_asset += pos_val

        lines.append(
            f"- {ticker}: 現値 {price_now:.1f} / 取得 {price_in:.1f} / 損益 {pnl_pct:+.2f}%"
        )

    if total_asset <= 0:
        total_asset = 2_000_000.0

    # 建玉 = total_pos / total_asset
    lev = total_pos / total_asset if total_asset > 0 else 1.0

    if not lines:
        return "ノーポジション", 2_000_000.0, 0.0, 1.0, {}

    text = "\n".join(lines)
    return text, float(total_asset), float(total_pos), float(lev), {}