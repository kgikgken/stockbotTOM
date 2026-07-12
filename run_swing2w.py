"""swing2w系のバックテスト用ラッパー。本番のswing2w/classify.py・indicators.py・screen.pyの
関数をそのまま呼び出す(バックテスト専用の判定ロジックを別途書かない)。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from swing2w.indicators import compute_swing_features, compute_sector_relative_z
from swing2w.classify import classify_engine_r, classify_engine_m
from .engine import Trade


def swing2w_signal_fn(ohlcv_full: Dict[str, pd.DataFrame], t_idx: int,
                      trading_dates: pd.DatetimeIndex, cfg) -> List[dict]:
    """T日時点でエンジンR/Mが新規に点灯した銘柄を返す。本番の回転率二分・
    compute_sector_relative_z・classify_engine_r/mをそのまま使う。"""
    today = trading_dates[t_idx]
    eligible = []
    for tkr, df_full in ohlcv_full.items():
        if today not in df_full.index:
            continue
        pos = df_full.index.get_loc(today)
        df = df_full.iloc[:pos + 1]
        if len(df) < 260:
            continue
        feat = compute_swing_features(df, cfg)
        if feat is None:
            continue
        if feat["close"] < cfg.min_price or feat["adv20_jpy"] < cfg.min_adv_jpy:
            continue
        row = {"ticker": tkr, "sector": df_full.attrs.get("sector", "不明"),
              "name": df_full.attrs.get("name", tkr)}
        eligible.append({"row": row, "feat": feat})

    if not eligible:
        return []

    advs = sorted(it["feat"]["adv20_jpy"] for it in eligible)
    low_cut = advs[int(len(advs) * cfg.turnover_low_pct)]
    high_cut = advs[min(int(len(advs) * cfg.turnover_high_pct), len(advs) - 1)]
    low_turnover = [it for it in eligible if it["feat"]["adv20_jpy"] <= low_cut]
    high_turnover = [it for it in eligible if it["feat"]["adv20_jpy"] >= high_cut]
    sector_z = compute_sector_relative_z(eligible, cfg)

    signals = []
    for item in low_turnover:
        c = classify_engine_r(item, sector_z, cfg)
        if c is not None:
            signals.append({"ticker": c.ticker, "engine": "R", "entry": c.entry, "stop": c.stop,
                            "target": c.target, "sector": c.sector, "name": c.name})
    for item in high_turnover:
        c = classify_engine_m(item, cfg)
        if c is not None:
            signals.append({"ticker": c.ticker, "engine": "M", "entry": c.entry, "stop": c.stop,
                            "target": c.target, "sector": c.sector, "name": c.name})
    return signals


def swing2w_exit_fn(trade: Trade, df_upto: pd.DataFrame, t_idx: int, time_stop_days: int = 10) -> Optional[tuple]:
    """保有中ポジションの手仕舞い判定。固定利確/初期ストップ/時間ストップのいずれか先着で手仕舞い
    (本番のswing2w/position_check.pyと同じロジックをバックテスト向けに再現)。
    time_stop_daysはcfg.time_stop_daysと必ず同じ値をfunctools.partialで固定して渡すこと。"""
    if len(df_upto) < 1:
        return None
    h, l, c = df_upto["High"], df_upto["Low"], df_upto["Close"]
    today_high, today_low, today_close = float(h.iloc[-1]), float(l.iloc[-1]), float(c.iloc[-1])

    if trade.target is not None and today_high >= trade.target:
        return trade.target, "target"
    if today_low <= trade.stop:
        return trade.stop, "stop"

    entry_dt = pd.Timestamp(trade.entry_date)
    today_dt = df_upto.index[-1]
    days_held = int(np.busday_count(entry_dt.date(), today_dt.date()))
    if days_held >= time_stop_days:
        return today_close, "time"
    return None
