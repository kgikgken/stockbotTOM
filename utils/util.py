from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

def jst_today_str():
    return datetime.now(JST).strftime("%Y-%m-%d")

def jst_today_date():
    return datetime.now(JST).date()

def parse_event_datetime_jst(*args):
    return None