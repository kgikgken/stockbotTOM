# ============================================
# utils/events.py
# マクロ・決算イベント管理
# ============================================

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict

from utils.util import jst_now, jst_today_str


# --------------------------------------------
# 設定
# --------------------------------------------
EARNINGS_BLOCK_DAYS = 3   # 決算前後◯営業日 新規禁止
MACRO_WARNING_DAYS = 2    # 重要イベント◯日前から警戒


# --------------------------------------------
# 決算チェック
# --------------------------------------------
def is_near_earnings(earnings_date: str) -> bool:
    """
    決算日前後の新規禁止判定
    earnings_date: "YYYY-MM-DD"
    """
    if not earnings_date:
        return False

    try:
        ed = datetime.strptime(earnings_date, "%Y-%m-%d").date()
    except Exception:
        return False

    today = jst_now().date()
    return abs((ed - today).days) <= EARNINGS_BLOCK_DAYS


# --------------------------------------------
# マクロイベント警戒
# --------------------------------------------
def macro_warning(events: List[Dict]) -> bool:
    """
    重要マクロイベントが近いか
    events = [{"name": "...", "date": "YYYY-MM-DD"}]
    """
    today = jst_now().date()

    for ev in events:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue

        if 0 <= (d - today).days <= MACRO_WARNING_DAYS:
            return True

    return False


# --------------------------------------------
# 表示用イベント文言
# --------------------------------------------
def format_events(events: List[Dict]) -> List[str]:
    """
    レポート用のイベント表記
    """
    out = []
    today = jst_today_str()

    for ev in events:
        name = ev.get("name", "")
        date = ev.get("date", "")
        if not name or not date:
            continue

        try:
            d = datetime.strptime(date, "%Y-%m-%d").date()
            days = (d - jst_now().date()).days
        except Exception:
            continue

        if days >= 0:
            out.append(f"⚠ {name}（{date} / {days}日後）")

    return out