from datetime import datetime, timedelta, timezone

# JST
JST = timezone(timedelta(hours=9))

def jst_today_str() -> str:
    """今日の日付（JST）を YYYY-MM-DD 形式で返す"""
    return datetime.now(JST).strftime("%Y-%m-%d")