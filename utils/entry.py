from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.features import Features
from utils.setup import SetupResult

@dataclass(frozen=True)
class EntryPlan:
    in_center: float
    in_low: float
    in_high: float
    gu_flag: bool
    in_distance_atr: float
    action: str

def calc_entry(hist: pd.DataFrame, f: Features, s: SetupResult) -> Optional[EntryPlan]:
    if hist is None or len(hist) < 30:
        return None
    close = hist["Close"].astype(float)
    open_ = hist["Open"].astype(float)
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else float(close.iloc[-1])
    today_open = float(open_.iloc[-1]) if len(open_) else float(close.iloc[-1])

    atr = float(f.atr)
    if not (np.isfinite(atr) and atr > 0):
        return None

    if s.setup == "A":
        center = float(f.ma20)
        band = 0.5 * atr
    else:
        center = float(s.breakout_line)
        band = 0.3 * atr

    in_low = center - band
    in_high = center + band

    gu_flag = bool(np.isfinite(today_open) and np.isfinite(prev_close) and (today_open > prev_close + 1.0 * atr))

    price = float(f.price)
    in_distance_atr = float(abs(price - center) / atr) if atr > 0 else 999.0

    if gu_flag or in_distance_atr > 0.8:
        action = "WATCH_ONLY"
    else:
        action = "EXEC_NOW" if (in_low <= price <= in_high) else "LIMIT_WAIT"

    return EntryPlan(
        in_center=float(center),
        in_low=float(in_low),
        in_high=float(in_high),
        gu_flag=gu_flag,
        in_distance_atr=float(in_distance_atr),
        action=action,
    )
