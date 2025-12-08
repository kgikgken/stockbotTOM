from __future__ import annotations
import pandas as pd
import yfinance as yf
import numpy as np
import os

# ============================================================
# 設定
# ============================================================
# 1セクターあたり何銘柄を見るか（重くなりすぎないよう制限）
MAX_TICKERS_PER_SECTOR = 20

# universeファイル（親コードの main.py と同じ）
UNIVERSE_PATH = "universe_jpx.csv"


# ============================================================
# セクター別：上昇率（5日）上位を返す
# ============================================================
def top_sectors_5d() -> list[tuple[str, float]]:
    """
    universe_jpx.csv を見て、各セクターの代表銘柄を拾い、
    5日騰落率を算出 → 上位順に返す。
    
    戻り値例:
        [("銀行", 3.4), ("電気機器", 1.9), ...]
    """
    if not os.path.exists(UNIVERSE_PATH):
        return []

    try:
        df = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    # sector（または industry_big）で分類
    if "sector" in df.columns:
        sec_col = "sector"
    elif "industry_big" in df.columns:
        sec_col = "industry_big"
    else:
        return []

    # 各セクターごとに代表銘柄を抽出
    sectors = []
    for name, sub in df.groupby(sec_col):
        tickers = sub["ticker"].astype(str).tolist()
        if not tickers:
            continue

        # 上位 MAX_TICKERS_PER_SECTOR だけ使う（無制限だと重すぎる）
        tickers = tickers[:MAX_TICKERS_PER_SECTOR]

        # セクター平均騰落率を算出
        chgs = []
        for t in tickers:
            chg = _fetch_change_5d(t)
            if np.isfinite(chg):
                chgs.append(chg)

        if chgs:
            avg_chg = float(np.mean(chgs))
            sectors.append((name, avg_chg))

    # 上位順に並べる
    sectors.sort(key=lambda x: x[1], reverse=True)
    return sectors


# ============================================================
# 内部：5日騰落計算
# ============================================================
def _fetch_change_5d(ticker: str) -> float | float("nan"):
    """
    ticker の5日騰落率（％）
    
    計算:
        (今日のClose / 5日前のClose - 1) * 100
    """
    try:
        df = yf.Ticker(ticker).history(period="6d")
        if df is None or len(df) < 5:
            return np.nan
        close = df["Close"].astype(float)
        return float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
    except Exception:
        return np.nan