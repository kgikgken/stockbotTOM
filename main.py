from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import score_stock, calc_in_rank, trend_gate
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions
from utils.qualify import runner_strength, runner_class, al3_score

# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SECTOR_TOP_N = 5
EARNINGS_EXCLUDE_DAYS = 3

SWING_TOP_N_DISPLAY = 5
SWING_SCORE_MIN = 72.0
SWING_RR_MIN = 2.0
SWING_EV_R_MIN = 0.40
REQUIRE_TREND_GATE = True
REQUIRE_RUNNER_MIN = 70.0
AL3_ONLY = True

def main():
    print("stockbotTOM main restored")

if __name__ == "__main__":
    main()
