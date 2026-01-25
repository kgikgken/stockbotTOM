from dataclasses import dataclass

@dataclass
class SetupInfo:
    ticker: str
    setup_type: str
    entry_price: float
    stop: float
    tp1: float
    rr: float
    rr_min: float
    ev_min: float
    rday_min: float
    exp_r_tp1: float
    hit_prob: float
    exp_days: float
    r_per_day: float
