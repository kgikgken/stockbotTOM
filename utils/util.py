from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


JST = timezone(timedelta(hours=9))
WEEKLY_STATE_PATH = "weekly_state.json"


def jst_now() -> datetime:
    return datetime.now(JST)


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date():
    return jst_now().date()


def safe_float(x, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def sleep_brief(sec: float = 0.25) -> None:
    try:
        time.sleep(sec)
    except Exception:
        pass


def fetch_history(ticker: str, period: str = "260d", auto_adjust: bool = True) -> Optional[pd.DataFrame]:
    for _ in range(3):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=auto_adjust)
            if df is not None and not df.empty:
                return df
        except Exception:
            sleep_brief(0.4)
    return None


def fetch_fast_price(ticker: str) -> float:
    df = fetch_history(ticker, period="10d")
    if df is None or df.empty:
        return float("nan")
    return safe_float(df["Close"].iloc[-1])


def parse_event_datetime_jst(dt_str: str | None, date_str: str | None, time_str: str | None) -> Optional[datetime]:
    dt_str = (dt_str or "").strip()
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip()

    if dt_str:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(dt_str, fmt).replace(tzinfo=JST)
            except Exception:
                pass

    if date_str and time_str:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(f"{date_str} {time_str}", fmt).replace(tzinfo=JST)
            except Exception:
                pass

    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=JST)
        except Exception:
            return None

    return None


# -----------------------------
# Weekly new-trade limit state
# -----------------------------
def _iso_year_week(d) -> Tuple[int, int]:
    iso = d.isocalendar()
    return int(iso.year), int(iso.week)


def load_weekly_state(path: str = WEEKLY_STATE_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"year": 0, "week": 0, "count": 0, "last_date": ""}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("bad state")
        obj.setdefault("year", 0)
        obj.setdefault("week", 0)
        obj.setdefault("count", 0)
        obj.setdefault("last_date", "")
        return obj
    except Exception:
        return {"year": 0, "week": 0, "count": 0, "last_date": ""}


def save_weekly_state(state: Dict[str, Any], path: str = WEEKLY_STATE_PATH) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        # 失敗しても落とさない（通知はレポートで行う）
        pass


@dataclass
class WeeklyLimitResult:
    weekly_count: int
    weekly_cap: int
    weekly_block: bool
    updated: bool
    note: str


def apply_weekly_new_trade_limit(
    today_date,
    baseline_new_ok: bool,
    weekly_cap: int = 3,
    path: str = WEEKLY_STATE_PATH,
) -> WeeklyLimitResult:
    """
    baseline_new_ok=True の日だけ「新規可日」をカウント対象とする。
    ただし同日複数回実行で二重カウントしない（last_date）。
    cap到達後は baseline_new_ok でも weekly_block=True を返す（新規禁止）。
    """
    state = load_weekly_state(path)
    y, w = _iso_year_week(today_date)

    # week rollover
    if int(state.get("year", 0)) != y or int(state.get("week", 0)) != w:
        state = {"year": y, "week": w, "count": 0, "last_date": ""}

    cnt = int(state.get("count", 0))
    last_date = str(state.get("last_date", ""))

    if not baseline_new_ok:
        return WeeklyLimitResult(cnt, weekly_cap, False, False, "週次カウント対象外（NO-TRADE/見送り）")

    if cnt >= weekly_cap:
        return WeeklyLimitResult(cnt, weekly_cap, True, False, "週次制限により新規停止")

    # increment once per day
    today_str = str(today_date)
    if last_date != today_str:
        cnt += 1
        state["count"] = cnt
        state["last_date"] = today_str
        save_weekly_state(state, path)
        return WeeklyLimitResult(cnt, weekly_cap, False, True, "週次カウント+1")

    return WeeklyLimitResult(cnt, weekly_cap, False, False, "本日は既に週次カウント済み")