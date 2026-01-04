from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from utils.util import clamp

@dataclass
class RRResult:
    stop: float
    tp1: float
    tp2: float
    rr: float
    pwin: float
    ev: float
    adj_ev: float
    exp_days: float
    r_per_day: float

def compute_rr_ev(in_price, stop, atr, cfg):
    risk = in_price - stop
    tp2 = in_price + cfg.TP2_R * risk
    rr = (tp2 - in_price) / risk
    exp_days = (tp2 - in_price) / atr
    r_day = rr / exp_days
    return rr, exp_days, r_day