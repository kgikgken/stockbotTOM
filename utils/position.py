# ============================================
# utils/position.py
# 既存ポジションの読み込み・リスク分析
# ============================================

from __future__ import annotations

import os
from typing import Dict, List
import pandas as pd


# --------------------------------------------
# ポジション読み込み
# --------------------------------------------
def load_positions(path: str = "positions.csv") -> pd.DataFrame:
    """
    positions.csv を読み込む
    想定カラム例:
      - ticker
      - qty
      - entry_price
      - stop_price
      - current_price（無くてもOK）
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


# --------------------------------------------
# ポジション分析
# --------------------------------------------
def analyze_positions(
    positions_df: pd.DataFrame,
    capital: float | None = None,
    risk_per_trade: float = 0.015,
) -> Dict:
    """
    保有ポジションのリスクを集計する

    戻り値:
      {
        "num_positions": int,
        "estimated_max_loss": float,
        "risk_ratio": float | None
      }
    """
    if positions_df is None or len(positions_df) == 0:
        return {
            "num_positions": 0,
            "estimated_max_loss": 0.0,
            "risk_ratio": 0.0 if capital else None,
        }

    total_loss = 0.0

    for _, row in positions_df.iterrows():
        try:
            qty = float(row.get("qty", 0))
            entry = float(row.get("entry_price", 0))
            stop = float(row.get("stop_price", entry))

            loss_per_share = max(entry - stop, 0)
            total_loss += loss_per_share * qty
        except Exception:
            continue

    risk_ratio = None
    if capital and capital > 0:
        risk_ratio = total_loss / capital

    return {
        "num_positions": int(len(positions_df)),
        "estimated_max_loss": float(total_loss),
        "risk_ratio": risk_ratio,
    }