# utils/sector.py
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import yfinance as yf


def top_sectors_5d() -> List[Tuple[str, float]]:
    """
    ここは簡易版：
    - TOPIX-33等の「セクター指数」を手元で持っていない前提
    - 代替として、TOPIXの業種別ETF/指数があるなら差し替え推奨
    現状は「空でも落とさない」扱い。
    """
    # 実運用ではここをあなたの sector.csv 等に置き換えてOK
    return []