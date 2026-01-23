from __future__ import annotations

import os
import datetime as dt
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional

import pandas as pd

JST = ZoneInfo("Asia/Tokyo")


def _parse_dt_jst(x: str) -> Optional[dt.datetime]:
    if not isinstance(x, str) or not x.strip():
        return None
    x = x.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            d = dt.datetime.strptime(x, fmt)
            if fmt in ("%Y-%m-%d", "%Y/%m/%d"):
                d = d.replace(hour=0, minute=0)
            return d.replace(tzinfo=JST)
        except Exception:
            pass
    try:
        d = dt.datetime.fromisoformat(x)
        if d.tzinfo is None:
            d = d.replace(tzinfo=JST)
        return d.astimezone(JST)
    except Exception:
        return None


def load_events(csv_path: str = "events.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["name", "datetime_jst", "importance"])
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame(columns=["name", "datetime_jst", "importance"])

    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("event") or df.columns[0]
    dt_col = cols.get("datetime_jst") or cols.get("datetime") or cols.get("date") or df.columns[1]
    imp_col = cols.get("importance") or cols.get("level")

    out = pd.DataFrame()
    out["name"] = df[name_col].astype(str)
    out["datetime_jst"] = df[dt_col].astype(str)
    out["importance"] = df[imp_col].astype(str) if imp_col in df.columns else "HIGH"

    out["dt"] = out["datetime_jst"].apply(_parse_dt_jst)
    out = out.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    return out


def upcoming_important(events_df: pd.DataFrame, now_jst: dt.datetime, horizon_days: int = 2) -> List[Dict]:
    if events_df is None or events_df.empty:
        return []
    end = now_jst + dt.timedelta(days=horizon_days)
    sub = events_df[(events_df["dt"] >= now_jst) & (events_df["dt"] <= end)]
    items: List[Dict] = []
    for _, r in sub.iterrows():
        items.append({"name": str(r["name"]), "dt": r["dt"]})
    return items
