from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# 日付（JST）
# ============================================================
def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date():
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
# DataFrame安全ラップ
# ============================================================
def ensure_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame()


# ============================================================
# 有効値平均
# ============================================================
def mean_valid(values, default=np.nan) -> float:
    try:
        arr = [float(x) for x in values if np.isfinite(float(x))]
        if not arr:
            return float(default)
        return float(np.mean(arr))
    except Exception:
        return float(default)