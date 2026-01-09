from __future__ import annotations

from datetime import date
from typing import Dict, List

import pandas as pd

from utils.util import parse_event_datetime_jst, business_days_between


# Macro判定：kind=="macro" もしくは label に以下が含まれる
MACRO_KEYWORDS = ("FOMC", "雇用", "雇用統計", "CPI", "PCE", "日銀", "BOJ", "ECB", "GDP", "失業", "ISM")


def load_events(path: str) -> List[Dict[str, str]]:
    if not path:
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    out: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        out.append(
            {
                "label": label,
                "kind": str(row.get("kind", "")).strip(),
                "date": str(row.get("date", "")).strip(),
                "time": str(row.get("time", "")).strip(),
                "datetime": str(row.get("datetime", "")).strip(),
            }
        )
    return out


def is_macro_caution(today_date: date, events: List[Dict[str, str]]) -> bool:
    """
    Macro警戒ON：today±2日 に macroイベントが存在
    """
    for ev in events or []:
        kind = (ev.get("kind") or "").strip().lower()
        label = (ev.get("label") or "").strip()
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        delta = (dt.date() - today_date).days
        is_macro = (kind == "macro") or any(k in label for k in MACRO_KEYWORDS)
        if is_macro and (-1 <= delta <= 2):
            return True
    return False


def build_event_warnings(today_date: date, events: List[Dict[str, str]]) -> List[str]:
    warns: List[str] = []
    for ev in events or []:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        delta = (dt.date() - today_date).days
        if -1 <= delta <= 2:
            when = "直近" if delta < 0 else ("本日" if delta == 0 else f"{delta}日後")
            dt_disp = dt.strftime("%Y-%m-%d %H:%M JST")
            warns.append(f"⚠ {ev.get('label','')}（{dt_disp} / {when}）")
    if not warns:
        warns.append("- 特になし")
    return warns


def earnings_new_entry_block(earnings_date_str: str | None, today_date: date, window_bd: int = 3) -> bool:
    """
    決算日 ±N営業日：新規禁止
    """
    if not earnings_date_str:
        return False
    try:
        d = pd.to_datetime(earnings_date_str, errors="coerce").date()
    except Exception:
        return False
    if d is None or pd.isna(d):
        return False
    bd = abs(business_days_between(today_date, d))
    return bd <= int(window_bd)
