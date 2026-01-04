from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class EntryPlan:
    setup_type: str
    in_low: float
    in_high: float
    in_center: float
    gu_flag: bool
    monitor_only: bool
    in_dist_atr: float