from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# 日付（JST）
# ============================================================

def jst_now() -> datetime:
    """JST 現在時刻"""
    return datetime.now(timezone(timedelta(hours=9)))


def jst_today_str() -> str:
    """今日の日付文字列 (YYYY-MM-DD JST)"""
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date():
    """今日の date (JST)"""
    return jst_now().date()


# ============================================================
# 安全な float 変換
# ============================================================

def safe_float(x, default: float = np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


# ============================================================
# DataFrame utilities
# ============================================================

def ensure_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """None などでも空 DataFrame を返す"""
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame()


def mean_valid(values, default: float = np.nan) -> float:
    """NaN を除外して平均を取る"""
    try:
        arr = [float(v) for v in values if np.isfinite(float(v))]
        if not arr:
            return float(default)
        return float(np.mean(arr))
    except Exception:
        return float(default)