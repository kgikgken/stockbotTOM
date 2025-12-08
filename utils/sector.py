from __future__ import annotations

import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = os.getenv("UNIVERSE_PATH", "universe_jpx.csv")

# 1セクターあたり何銘柄まで見るか（重くなりすぎないよう制限）
MAX_TICKERS_PER_SECTOR = 20


# ============================================================
# ユニバース読み込み（セクター用）
# ============================================================
def _load_universe_for_sector(path: str = UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    """
    universe_jpx.csv からセクター情報だけ使う。
    必要カラム:
      - ticker
      - sector または industry_big
    """
    if not os.path.exists(path):
        print(f"[sector] universe file not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[sector] failed to read universe: {e}")
        return None

    if "ticker" not in df.columns:
        print("[sector] universe has no 'ticker' column")
        return None

    # セクター列を決定
    if "sector" in df.columns:
        sector_col = "sector"
    elif "industry_big" in df.columns:
        sector_col = "industry_big"
    else:
        print("[sector] universe has no 'sector' or 'industry_big' column")
        return None

    df = df[["ticker", sector_col]].copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["sector"] = df[sector_col].astype(str).str.strip().replace({"": "不明", "nan": "不明"})

    df = df[df["ticker"] != ""]
    return df


# ============================================================
# 単銘柄 5日騰落率
# ============================================================
def _five_day_change_pct(ticker: str) -> Optional[float]:
    """
    1銘柄の5日騰落率（%）を返す。
    データ不足やエラー時は None。
    """
    try:
        hist = yf.Ticker(ticker).history(period="6d")
        if hist is None or hist.empty or len(hist) < 2:
            return None

        close = hist["Close"].astype(float)
        first = float(close.iloc[0])
        last = float(close.iloc[-1])
        if first <= 0:
            return None

        return (last / first - 1.0) * 100.0
    except Exception:
        return None


# ============================================================
# セクター別 5日騰落率
# ============================================================
def _calc_sector_changes(df_uni: pd.DataFrame) -> Dict[str, float]:
    """
    セクターごとに構成銘柄の 5 日騰落率平均を計算。
    戻り値: { "銀行": 2.34, "機械": -0.85, ... } （単位: %）
    """
    sector_ret: Dict[str, float] = {}

    grouped = df_uni.groupby("sector")

    for sector, g in grouped:
        tickers = list(g["ticker"].unique())
        if not tickers:
            continue

        # 重くなりすぎないように上限
        if len(tickers) > MAX_TICKERS_PER_SECTOR:
            tickers = tickers[:MAX_TICKERS_PER_SECTOR]

        rets: List[float] = []
        for t in tickers:
            r = _five_day_change_pct(t)
            if r is not None and np.isfinite(r):
                rets.append(float(r))

        if not rets:
            continue

        avg_ret = float(np.mean(rets))
        sector_ret[sector] = avg_ret

    return sector_ret


# ============================================================
# 公開 API: top_sectors_5d
# ============================================================
def top_sectors_5d(top_n: int = 5) -> List[Tuple[str, float]]:
    """
    セクター別 5 日騰落率ランキングを返す。

    戻り値: [("銀行", +3.01), ("機械", +1.29), ...]
      - 降順ソート済み
      - top_n 件だけ返す（デフォルト 5）
    データ取得に失敗したら []。
    """
    df_uni = _load_universe_for_sector()
    if df_uni is None or df_uni.empty:
        return []

    sector_ret = _calc_sector_changes(df_uni)
    if not sector_ret:
        return []

    items = sorted(sector_ret.items(), key=lambda x: x[1], reverse=True)
    if top_n > 0:
        items = items[:top_n]

    # 小数 2 桁に丸めて返す
    return [(name, round(chg, 2)) for name, chg in items]