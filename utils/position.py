from __future__ import annotations

import os
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf

from utils import rr


def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    """
    現在ポジション一覧を読み込む。
    無い場合は空DataFrameを返す。
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ticker", "size", "avg_price"])
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read positions: {e}")
        return pd.DataFrame(columns=["ticker", "size", "avg_price"])

    if "ticker" not in df.columns:
        df["ticker"] = ""
    if "size" not in df.columns:
        df["size"] = 0
    if "avg_price" not in df.columns:
        df["avg_price"] = np.nan

    return df


def _fetch_last_price(ticker: str) -> float:
    """
    現値を取得（5日・日足）。
    """
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return np.nan
        return float(df["Close"].iloc[-1])
    except Exception as e:
        print(f"[WARN] price fetch failed for {ticker}: {e}")
        return np.nan


def analyze_positions(
    df: pd.DataFrame,
) -> Tuple[str, float, float, float, Dict]:
    """
    ポジションサマリ文字列と、推定総資産などを返す。

    戻り値:
      pos_text: 表示用テキスト
      total_asset: 推定総資産
      total_pos: 建玉総額
      lev: レバレッジ
      risk_info: 予備情報(dict)
    """
    if df is None or df.empty:
        pos_text = "ノーポジション"
        total_asset = 2_000_000.0
        total_pos = 0.0
        lev = 0.0
        risk_info: Dict = {}
        return pos_text, total_asset, total_pos, lev, risk_info

    rows: List[str] = []
    total_pos = 0.0

    if "cash" in df.columns:
        try:
            cash = float(df["cash"].iloc[0])
        except Exception:
            cash = 0.0
    else:
        cash = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        size = float(row.get("size", 0) or 0)
        avg_price = float(row.get("avg_price", np.nan) or np.nan)

        if size <= 0:
            continue

        cur = _fetch_last_price(ticker)
        if not np.isfinite(cur):
            continue

        pnl_pct = (cur - avg_price) / avg_price * 100.0 if avg_price > 0 else 0.0
        pos_val = cur * size
        total_pos += pos_val

        rows.append(
            f"- {ticker}: 現値 {cur:.1f} / 取得 {avg_price:.1f} / 損益 {pnl_pct:.2f}%"
        )

    total_asset = total_pos + cash if total_pos + cash > 0 else 2_000_000.0
    lev = total_pos / total_asset if total_asset > 0 else 0.0

    pos_text = "\n".join(rows) if rows else "ノーポジション"
    risk_info: Dict = {
        "cash": cash,
        "total_pos": total_pos,
        "lev": lev,
    }
    return pos_text, total_asset, total_pos, lev, risk_info


def compute_positions_rr(
    df: pd.DataFrame,
    mkt_score: int,
) -> Dict[str, Dict[str, float]]:
    """
    各保有銘柄の「今から見たRR」を計算する。
    RRは「今の位置からTP/SLを取った場合の期待R」。

    戻り値:
      { "ticker": {"rr": rr_value} }
    """
    result: Dict[str, Dict[str, float]] = {}

    if df is None or df.empty:
        return result

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        try:
            hist = yf.Ticker(ticker).history(period="130d")
            if hist is None or hist.empty:
                continue
        except Exception as e:
            print(f"[WARN] history fetch failed for {ticker}: {e}")
            continue

        try:
            info = rr.compute_tp_sl_rr(hist, mkt_score)
        except Exception as e:
            print(f"[WARN] RR calc failed for {ticker}: {e}")
            continue

        rr_val = float(info.get("rr", 0.0))
        if not np.isfinite(rr_val):
            continue

        result[ticker] = {"rr": rr_val}

    return result