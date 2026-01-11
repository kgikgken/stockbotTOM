from __future__ import annotations

from typing import List, Dict, Any
import os
import pandas as pd

from utils.util import parse_event_datetime_jst, business_days_diff

MACRO_KINDS = {
    "FOMC", "雇用統計", "日銀", "BOJ", "CPI", "PCE", "GDP", "ISM", "金融政策", "政策金利"
}

def load_events(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
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
        out.append({
            "label": label,
            "kind": str(row.get("kind", "")).strip(),
            "date": str(row.get("date", "")).strip(),
            "time": str(row.get("time", "")).strip(),
            "datetime": str(row.get("datetime", "")).strip(),
        })
    return out

def build_event_items(today_date, events: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for ev in events or []:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        delta_days = (dt.date() - today_date).days
        if -1 <= delta_days <= 2:
            when = "直近" if delta_days < 0 else ("本日" if delta_days == 0 else f"{delta_days}日後")
            items.append({
                "label": str(ev.get("label", "")).strip(),
                "kind": str(ev.get("kind", "")).strip(),
                "dt": dt,
                "dt_str": dt.strftime("%Y-%m-%d %H:%M JST"),
                "when": when,
            })
    items.sort(key=lambda x: x["dt"])
    return items

def is_macro_caution(today_date, events: List[Dict[str, str]]) -> bool:
    for ev in events or []:
        dt = parse_event_datetime_jst(ev.get("datetime"), ev.get("date"), ev.get("time"))
        if dt is None:
            continue
        bd = business_days_diff(today_date, dt.date())
        if -1 <= bd <= 2:
            kind = str(ev.get("kind", "")).strip()
            label = str(ev.get("label", "")).strip()
            if kind in MACRO_KINDS:
                return True
            for k in MACRO_KINDS:
                if k and k in label:
                    return True
    return False
