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


def jst_today_date() -> datetime.date:
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
        return v
    except Exception:
        return float(default)


# ============================================================
# データフレームのチェック
# ============================================================

def ensure_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """None や invalid の場合でも空DF返す"""
    if df is None:
        return pd.DataFrame()
    try:
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    return pd.DataFrame()


# ============================================================
# 平均（NaN 全除去）
# ============================================================

def mean_valid(values, default=np.nan) -> float:
    """NaNなど無視して平均"""
    try:
        arr = [float(x) for x in values if np.isfinite(float(x))]
        if len(arr) == 0:
            return float(default)
        return float(np.mean(arr))
    except Exception:
        return float(default)