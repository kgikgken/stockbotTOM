# ============================================
# utils/position.py
# 保有ポジション管理・リスク集計
# ============================================

from __future__ import annotations

import pandas as pd
from typing import Dict, Tuple


# --------------------------------------------
# ポジション読み込み
# --------------------------------------------
def load_positions(path: str) -> pd.DataFrame:
    """
    positions.csv を読み込む
    想定カラム:
      ticker, entry, size, stop, rr
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    # 必須カラム補正
    for col in ["ticker", "entry", "size", "stop", "rr"]:
        if col not in df.columns:
            df[col] = None

    return df


# --------------------------------------------
# ポジション分析
# --------------------------------------------
def analyze_positions(
    pos_df: pd.DataFrame,
    capital: float,
) -> Dict:
    """
    現在ポジションのリスク状況を集計
    """
    if pos_df is None or pos_df.empty:
        return {
            "count": 0,
            "max_loss": 0.0,
            "risk_ratio": 0.0,
            "warning": False,
        }

    max_loss = 0.0

    for _, row in pos_df.iterrows():
        try:
            entry = float(row["entry"])
            stop = float(row["stop"])
            size = float(row["size"])
            loss = max(entry - stop, 0) * size
            max_loss += loss
        except Exception:
            continue

    risk_ratio = max_loss / capital if capital > 0 else 0.0

    return {
        "count": len(pos_df),
        "max_loss": round(max_loss, 0),
        "risk_ratio": round(risk_ratio, 4),
        "warning": risk_ratio >= 0.10,  # 10%超で警告
    }


# --------------------------------------------
# セクター・相関制限チェック用
# --------------------------------------------
def sector_exposure(pos_df: pd.DataFrame) -> Dict[str, int]:
    """
    セクター別保有数
    """
    if pos_df is None or pos_df.empty:
        return {}

    if "sector" not in pos_df.columns:
        return {}

    return pos_df["sector"].value_counts().to_dict()


def can_add_position(
    pos_df: pd.DataFrame,
    sector: str,
    max_per_sector: int = 2,
) -> bool:
    """
    セクター上限チェック
    """
    if pos_df is None or pos_df.empty:
        return True

    if "sector" not in pos_df.columns:
        return True

    current = (pos_df["sector"] == sector).sum()
    return current < max_per_sector