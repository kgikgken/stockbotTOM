from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Paths
    UNIVERSE_PATH: str = "universe_jpx.csv"
    POSITIONS_PATH: str = "positions.csv"
    EVENTS_PATH: str = "events.csv"

    # Index
    TOPIX_TICKER: str = "998405.T"

    # Universe
    PRICE_MIN: float = 200.0
    PRICE_MAX: float = 15000.0
    ADV20_MIN_JPY: float = 100_000_000.0
    ATRPCT_MIN: float = 0.015

    # Earnings
    EARNINGS_EXCLUDE_DAYS: int = 3

    # Output sizes
    MAX_FINAL_STOCKS: int = 5
    MAX_WATCHLIST: int = 10
    SECTOR_TOP_K: int = 5

    # NO-TRADE
    NO_TRADE_MKT_SCORE_LT: float = 45.0
    NO_TRADE_MKT_SCORE_LT_WHEN_MOM_DOWN: float = 55.0
    NO_TRADE_MOMENTUM_3D_LTE: float = -5.0
    NO_TRADE_AVG_ADJEV_LT_R: float = 0.3
    NO_TRADE_GU_RATIO_GTE: float = 0.60

    # Entry
    GU_ATR_MULT: float = 1.0
    IN_PULLBACK_ATR_HALF_BAND: float = 0.5
    IN_BREAKOUT_ATR_BAND: float = 0.3
    IN_DIST_MONITOR_ATR: float = 0.8

    # Stops / Targets
    STOP_PULLBACK_EXTRA_ATR: float = 0.7
    STOP_BREAKOUT_ATR: float = 1.0
    TP1_R: float = 1.5
    TP2_R: float = 3.0

    # EV / RR
    RR_MIN: float = 2.2
    EV_MIN_R: float = 0.4
    EV_MIN_R_NEUTRAL: float = 0.5

    # Time efficiency
    EXP_DAYS_MAX: float = 5.0
    R_PER_DAY_MIN: float = 0.5
    EXP_DAYS_K_ATR: float = 1.0

    # Diversification
    MAX_PER_SECTOR: int = 2
    CORR_WINDOW: int = 20
    CORR_MAX: float = 0.75

    # Market
    MKT_SMA_FAST: int = 20
    MKT_SMA_SLOW: int = 50
    MKT_SMA_RISK: int = 10

    # Multipliers
    MULT_STRONG_UP: float = 1.05
    MULT_MOM_DOWN: float = 0.70
    MULT_EVENT_EVE: float = 0.75

    WORKER_TIMEOUT_SEC: int = 20


def load_config() -> Config:
    return Config()