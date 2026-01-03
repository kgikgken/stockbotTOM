from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from utils.features import FeaturePack


@dataclass
class EntryPlan:
    in_center: float
    band_low: float
    band_high: float
    dist_atr: float
    gu_flag: bool
    action: str  # EXEC_NOW / LIMIT_WAIT / WATCH_ONLY
    reason: str


def build_entry_plan(fp: FeaturePack, setup_type: str, hh20: float) -> EntryPlan:
    """
    仕様：
      A：IN_center=MA20、帯=±0.5ATR
      B：IN_center=HH20、帯=±0.3ATR
      GU_flag：Open > PrevClose + 1.0ATR（proxy）
      乖離率：distance(Close, IN_center)/ATR
      行動：
        EXEC_NOW：帯の中 & GUなし & 乖離小 &（追加）当日出来高<20日平均出来高
        LIMIT_WAIT：帯の外だが 0.8以内（指値）
        WATCH_ONLY：GU or 乖離>0.8
    """
    atr = fp.atr if np.isfinite(fp.atr) and fp.atr > 0 else max(fp.close * 0.01, 1.0)

    if setup_type == "B":
        in_center = float(hh20)
        w = 0.3 * atr
    else:
        in_center = float(fp.ma20) if np.isfinite(fp.ma20) else float(fp.close)
        w = 0.5 * atr

    band_low = in_center - w
    band_high = in_center + w

    dist_atr = abs(fp.close - in_center) / atr if atr > 0 else 999.0

    gu_flag = False
    if np.isfinite(fp.gap_atr):
        gu_flag = bool(fp.gap_atr >= 1.0)

    # basic action
    if gu_flag:
        return EntryPlan(in_center, band_low, band_high, float(dist_atr), True, "WATCH_ONLY", "GU危険域")

    if dist_atr > 0.8:
        return EntryPlan(in_center, band_low, band_high, float(dist_atr), False, "WATCH_ONLY", "乖離>0.8ATR")

    in_band = (band_low <= fp.close <= band_high)

    # EXEC_NOW 厳格化（出来高条件）
    vol_quiet_ok = False
    if np.isfinite(fp.vol_last) and np.isfinite(fp.vol_ma20) and fp.vol_ma20 > 0:
        vol_quiet_ok = bool(fp.vol_last < fp.vol_ma20)

    if in_band and vol_quiet_ok:
        return EntryPlan(in_center, band_low, band_high, float(dist_atr), False, "EXEC_NOW", "帯内+GUなし+出来高枯れ")

    # それ以外は基本 LIMIT_WAIT
    if in_band and not vol_quiet_ok:
        return EntryPlan(in_center, band_low, band_high, float(dist_atr), False, "LIMIT_WAIT", "帯内だが出来高が枯れてない")
    return EntryPlan(in_center, band_low, band_high, float(dist_atr), False, "LIMIT_WAIT", "帯外→指値待ち")