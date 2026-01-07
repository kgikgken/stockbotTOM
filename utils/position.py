# ============================================
# utils/position.py
# 保有ポジションの読み込み＆ロット事故警告
# ============================================

from __future__ import annotations

from typing import Dict, List, Tuple
import os
import pandas as pd


def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        return df
    except Exception:
        return pd.DataFrame()


def analyze_positions(df: pd.DataFrame, market_score: float, macro_risk: bool) -> Tuple[Dict, str]:
    """
    positions.csv 例（任意）:
      ticker, qty, avg_price, capital_jpy
    capital_jpy はどこか1行に入っていれば拾う
    """
    summary: Dict = {"capital_jpy": 2_000_000.0, "text": "n/a"}
    if df is None or df.empty:
        return summary, ""

    # capital_jpy
    if "capital_jpy" in df.columns:
        try:
            cap = df["capital_jpy"].dropna().astype(float)
            if len(cap) > 0:
                summary["capital_jpy"] = float(cap.iloc[0])
        except Exception:
            pass

    # 表示
    tickers = []
    if "ticker" in df.columns:
        tickers = [str(x) for x in df["ticker"].dropna().tolist()]

    if tickers:
        summary["text"] = "- " + "\n- ".join([f"{t}: 損益 n/a" for t in tickers])
    else:
        summary["text"] = "n/a"

    # ロット事故警告（簡易）
    # ここでは「最大建玉の一定割合が一度に被弾しうる」想定で出す（厳密には改善余地）
    cap = float(summary["capital_jpy"])
    # 危険度（地合い弱いほど警戒）
    risk_mult = 1.0
    if market_score < 50:
        risk_mult = 1.2
    if macro_risk:
        risk_mult *= 1.1

    # 仮：想定最大損失 = 資産 * 0.09 * risk_mult（出力だけ）
    worst = cap * 0.09 * risk_mult
    ratio = worst / cap * 100.0 if cap > 0 else 0.0

    lot_risk_text = ""
    if ratio >= 8.0:
        lot_risk_text = f"⚠ ロット事故警告：想定最大損失 ≈ {int(worst):,}円（資産比 {ratio:.2f}%）"

    return summary, lot_risk_text