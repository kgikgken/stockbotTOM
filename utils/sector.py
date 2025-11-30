import yfinance as yf
import pandas as pd
import numpy as np

# ============================================================
# TOPIX-17 セクター別ETF（代替）
# ============================================================
SECTOR_ETF = {
    "自動車": "1622.T",
    "機械": "1624.T",
    "電気機器": "1625.T",
    "精密機器": "1626.T",
    "銀行": "1615.T",
    "不動産": "1633.T",
    "小売": "1630.T",
    "建設": "1619.T",
    "鉄鋼": "1622.T",
    "化学": "1639.T",
    "サービス": "1627.T",
    "運輸": "1614.T",
}

def top_sectors_5d(top_n=3):
    results = []

    for sec, ticker in SECTOR_ETF.items():
        try:
            df = yf.Ticker(ticker).history(period="10d")
            if df is None or len(df) < 6:
                continue

            close = df["Close"]
            pct = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100
            results.append((sec, pct))
        except:
            continue

    if not results:
        return []

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
