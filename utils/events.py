# ============================================
# utils/events.py
# マクロ・イベント管理
# - 決算 / マクロイベントの検知
# - 新規可否・警戒フラグ生成
# ============================================

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd

from utils.util import jst_now, jst_today_str


# --------------------------------------------
# イベント定義
# --------------------------------------------
IMPORTANT_KEYWORDS = [
    "FOMC",
    "雇用統計",
    "CPI",
    "GDP",
    "日銀",
    "BOJ",
]


# --------------------------------------------
# events.csv 読み込み
# --------------------------------------------
def load_events(path: str) -> pd.DataFrame:
    """
    events.csv を読み込む
    必須列: date, name
    date は YYYY-MM-DD or YYYY-MM-DD HH:MM
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["date", "name"])

    if "date" not in df.columns or "name" not in df.columns:
        return pd.DataFrame(columns=["date", "name"])

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return df


# --------------------------------------------
# 重要イベント抽出
# --------------------------------------------
def extract_important_events(
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    IMPORTANT_KEYWORDS を含むイベントのみ抽出
    """
    if events_df.empty:
        return events_df

    mask = events_df["name"].apply(
        lambda x: any(k in str(x) for k in IMPORTANT_KEYWORDS)
    )
    return events_df[mask]


# --------------------------------------------
# 直近イベント判定
# --------------------------------------------
def check_upcoming_events(
    events_df: pd.DataFrame,
    days_ahead: int = 2,
) -> Dict[str, object]:
    """
    直近イベントを判定
    """
    now = jst_now()
    end = now + timedelta(days=days_ahead)

    upcoming = events_df[
        (events_df["date"] >= now) & (events_df["date"] <= end)
    ].sort_values("date")

    macro_risk = not upcoming.empty

    return {
        "macro_risk": macro_risk,
        "events": upcoming.to_dict("records"),
    }


# --------------------------------------------
# 決算回避判定
# --------------------------------------------
def is_near_earnings(
    earnings_date: Optional[str],
    window: int = 3,
) -> bool:
    """
    決算 ±window 日以内かどうか
    """
    if not earnings_date:
        return False

    try:
        ed = pd.to_datetime(earnings_date)
    except Exception:
        return False

    today = pd.to_datetime(jst_today_str())
    return abs((ed - today).days) <= window


# --------------------------------------------
# イベント制限ルール
# --------------------------------------------
def apply_event_constraints(
    macro_risk: bool,
    candidates: List[Dict],
    max_candidates_normal: int,
    max_candidates_event: int = 2,
) -> List[Dict]:
    """
    マクロイベント接近時は候補数を制限
    """
    if not macro_risk:
        return candidates[:max_candidates_normal]

    return candidates[:max_candidates_event]


# --------------------------------------------
# イベントサマリー文字列
# --------------------------------------------
def format_event_summary(events: List[Dict]) -> List[str]:
    """
    LINE 用イベント表示
    """
    lines = []
    for e in events:
        dt = e.get("date")
        name = e.get("name", "")
        if isinstance(dt, (datetime, pd.Timestamp)):
            dt_str = dt.strftime("%Y-%m-%d %H:%M")
        else:
            dt_str = str(dt)
        lines.append(f"{name}（{dt_str}）")
    return lines