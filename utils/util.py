
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))

def jst_now():
    return datetime.now(JST)

def jst_today():
    return jst_now().strftime("%Y-%m-%d")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def rr_min_by_market(score):
    if score >= 70: return 1.8
    if score >= 60: return 2.0
    if score >= 50: return 2.2
    return 2.5
