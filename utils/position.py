from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


# ============================================================
# positions.csv 読み込み
# ============================================================
def load_positions(path: str) -> pd.DataFrame:
    """
    positions.csv を読み込む。
    無い or 空 → 空 DataFrame。
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    # columns: ticker, shares, avg_price
    # 足りない列は補完
    for col in ["ticker", "shares", "avg_price"]:
        if col not in df.columns:
            df[col] = None

    df["ticker"] = df["ticker"].astype(str)
    return df


# ============================================================
# Yahooで現在価格に必要な ticker summary
# ============================================================
def fetch_last_price(ticker: str) -> float:
    """
    現値取得（エラー時は 0.0）
    """
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period="1d")
        if df is None or df.empty:
            return 0.0
        return float(df["Close"].iloc[-1])
    except Exception:
        return 0.0


# ============================================================
# 全ポジションの評価額・合計資産
# ============================================================
def analyze_positions(df: pd.DataFrame) -> Tuple[str, float, float, float, Dict]:
    """
    ポジション情報から
    - 日本語サマリ（LINE表示用テキスト）
    - 総資産推定
    - 総建玉
    - レバレッジ
    - リスク情報（dict）

    を返す。
    """
    # ノーポジ：初期資産2Mで返す
    if df is None or df.empty:
        text = "現在ポジションなし\n"
        total_asset = 2_000_000.0
        total_pos = 0.0
        lev = 1.0
        risk = {
            "positions": [],
            "cnt": 0,
            "loss_risk": 0.0,
            "gain_potential": 0.0,
        }
        return text, total_asset, total_pos, lev, risk

    lines: List[str] = []
    pos_values = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        shares = float(row.get("shares", 0))
        avg_price = float(row.get("avg_price", 0))

        if shares <= 0 or avg_price <= 0:
            continue

        cur = fetch_last_price(ticker)
        if cur <= 0:
            continue

        pnl = (cur / avg_price - 1.0) * 100.0
        val = cur * shares
        pos_values.append(val)

        lines.append(
            f"- {ticker}: 現値 {cur:.1f} / 取得 {avg_price:.1f} / 損益 {pnl:+.2f}%"
        )

    # もし全ポジが異常（取得0、現値0）ならノーポジ扱い
    if not pos_values:
        text = "現在ポジションなし\n"
        total_asset = 3_000_000.0
        total_pos = 0.0
        lev = 1.0
        risk = {
            "positions": [],
            "cnt": 0,
            "loss_risk": 0.0,
            "gain_potential": 0.0,
        }
        return text, total_asset, total_pos, lev, risk

    # 総建玉 = 各ポジション評価額の合計
    total_pos = float(np.sum(pos_values))

    # 総資産（単純に建玉=資産と考える）
    total_asset = total_pos

    # レバレッジ: 現状建玉/資産（現状は1.0固定に近い）
    lev = 1.0

    # 基本テキスト
    text = "\n".join(lines)

    # 簡易リスク情報（Futureで使う）
    risk = {
        "positions": [],         # Phase2で詳細入れる
        "cnt": len(pos_values),
        "loss_risk": 0.0,
        "gain_potential": 0.0,
    }

    return text, total_asset, total_pos, lev, risk