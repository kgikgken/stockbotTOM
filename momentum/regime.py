"""STEP1: レジームフィルター(TOPIX) — 例外なく厳守.

攻撃モード: TOPIX終値 > 200日移動平均 かつ 直近12ヶ月リターン > 0
それ以外は防御モード(新規エントリー全停止)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import fetch_regime_series


def compute_regime(cfg) -> dict:
    series, source = fetch_regime_series(cfg)
    if series is None or len(series) < cfg.regime_sma_days:
        return {
            "ok": False, "mode": "防御モード", "attack": False,
            "reason": f"レジーム系列取得不可({source}) — 安全側で防御モード",
            "source": source,
        }

    close_now = float(series.iloc[-1])
    sma200 = float(series.iloc[-cfg.regime_sma_days:].mean())
    mom_days = min(cfg.regime_mom_days, len(series) - 1)
    mom_12m = close_now / float(series.iloc[-1 - mom_days]) - 1.0

    above_sma = close_now > sma200
    positive_mom = mom_12m > 0
    attack = above_sma and positive_mom

    return {
        "ok": True,
        "attack": attack,
        "mode": "攻撃モード" if attack else "防御モード",
        "close": close_now, "sma200": sma200, "mom_12m": mom_12m * 100,
        "above_sma": above_sma, "positive_mom": positive_mom,
        "source": source,
        "detail": (f"TOPIX {close_now:,.1f} vs 200日線 {sma200:,.1f}"
                  f"({'上' if above_sma else '下'}) / 12ヶ月{mom_12m*100:+.1f}%"
                  f"({'プラス' if positive_mom else 'マイナス'})"),
    }
