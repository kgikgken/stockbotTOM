from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.features import Features

@dataclass(frozen=True)
class SetupResult:
    setup: str
    breakout_line: float
    reason: str

def detect_setup(hist: pd.DataFrame, f: Features) -> Optional[SetupResult]:
    price = f.price
    if not (np.isfinite(price) and price > 0):
        return None

    near = abs(price - f.ma20) <= 0.8 * f.atr
    a_ok = (
        np.isfinite(f.ma20) and np.isfinite(f.ma50) and
        price > f.ma20 > f.ma50 and
        np.isfinite(f.slope20_5d) and f.slope20_5d > 0 and
        np.isfinite(f.rsi) and 40 <= f.rsi <= 62 and
        near
    )
    if a_ok:
        return SetupResult(setup="A", breakout_line=float(f.ma20), reason="トレンド押し目")

    if hist is None or len(hist) < 60:
        return None
    vol = hist["Volume"].astype(float) if "Volume" in hist.columns else None
    vol_last = float(vol.iloc[-1]) if vol is not None and len(vol) else np.nan
    vol_ma20 = float(np.nan_to_num(f.vol_ma20, nan=0.0))
    b_ok = (
        np.isfinite(f.hh20) and price >= f.hh20 and
        np.isfinite(vol_last) and np.isfinite(vol_ma20) and vol_ma20 > 0 and
        vol_last >= 1.5 * vol_ma20
    )
    if b_ok:
        return SetupResult(setup="B", breakout_line=float(f.hh20), reason="ブレイク初動")

    return None
