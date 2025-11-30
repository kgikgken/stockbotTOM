import pandas as pd
import yfinance as yf
import numpy as np


# ===============================================
# 読み込み
# ===============================================
def load_positions(path="positions.csv"):
    try:
        df = pd.read_csv(path)
        if not {"ticker", "qty", "avg_price"}.issubset(df.columns):
            return None
        df["ticker"] = df["ticker"].astype(str)
        df["qty"] = df["qty"].astype(float)
        df["avg_price"] = df["avg_price"].astype(float)
        return df
    except:
        return None


# ===============================================
# 個別銘柄の株価取得（安全版）
# ===============================================
def safe_price(ticker: str):
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        if len(data) == 0:
            return None
        return float(data["Close"].iloc[-1])
    except:
        return None


# ===============================================
# ポジション分析（メイン処理）
# ===============================================
def analyze_positions(df_pos):
    """
    df_pos: positions.csv のDF
    戻り値: dict リスト
    """
    results = []
    total_value = 0

    for _, row in df_pos.iterrows():
        ticker = row["ticker"]
        qty = row["qty"]
        avg = row["avg_price"]

        price = safe_price(ticker)
        if price is None:
            results.append({
                "ticker": ticker,
                "qty": qty,
                "avg_price": avg,
                "price": None,
                "pnl_pct": None,
                "value": None,
            })
            continue

        value = price * qty
        pnl_pct = (price - avg) / avg * 100

        total_value += value

        results.append({
            "ticker": ticker,
            "qty": qty,
            "avg_price": avg,
            "price": price,
            "pnl_pct": pnl_pct,
            "value": value,
        })

    return results, total_value
