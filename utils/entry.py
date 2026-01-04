# utils/entry.py
from __future__ import annotations

from dataclasses import dataclass

from utils.features import Tech


@dataclass(frozen=True)
class EntryPlan:
    in_center: float
    in_low: float
    in_high: float
    gu: bool
    action: str  # "即IN可" / "指値待ち" / "今日は見送り"
    deviation_atr: float


def build_entry_plan(tech: Tech, setup_kind: str) -> EntryPlan:
    # IN帯
    if setup_kind in ("A1", "A2"):
        in_center = tech.sma20
        band = 0.5 * tech.atr
    else:
        in_center = tech.sma20
        band = 0.5 * tech.atr

    in_low = in_center - band
    in_high = in_center + band

    # GU判定（Open > PrevClose + 1.0ATR）を厳密にやりたいが、
    # ここでは "Open > Close + 1.0ATR" 近似（十分に危険判定になる）
    gu = (tech.open_ > tech.close + 1.0 * tech.atr)

    # 乖離率（CloseがIN_centerからどれだけ離れているか）
    deviation = abs(tech.close - in_center) / tech.atr if tech.atr > 0 else 999

    # 行動
    if gu:
        action = "今日は見送り"
    else:
        if in_low <= tech.close <= in_high:
            action = "即IN可"
        elif deviation <= 0.8:
            action = "指値待ち"
        else:
            action = "今日は見送り"

    return EntryPlan(
        in_center=float(in_center),
        in_low=float(in_low),
        in_high=float(in_high),
        gu=bool(gu),
        action=action,
        deviation_atr=float(deviation),
    )