from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.util import download_history_bulk, safe_float, is_abnormal_stock
from utils.setup import build_setup_info, liquidity_filters
from utils.rr_ev import calc_ev
from utils.diversify import apply_sector_cap, apply_corr_filter
from utils.screen_logic import no_trade_conditions, max_display
from utils.state import (
    in_cooldown,
    set_cooldown_days,
    record_paper_trade,
    update_paper_trades_with_ohlc,
    kpi_distortion,
)

UNIVERSE_PATH = "universe_jpx.csv"
EARNINGS_EXCLUDE_DAYS = 3  # 暦日近似（±3日）

def _get_ticker_col(df: pd.DataFrame) -> str:
    if "ticker" in df.columns:
        return "ticker"
    if "code" in df.columns:
        return "code"
    return ""

def _filter_earnings(uni: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in uni.columns:
        return uni
    d = pd.to_datetime(uni["earnings_date"], errors="coerce").dt.date
    uni = uni.copy()
    keep = []
    for x in d:
        if x is None or pd.isna(x):
            keep.append(True)
            continue
        try:
            keep.append(abs((x - today_date).days) > EARNINGS_EXCLUDE_DAYS)
        except Exception:
            keep.append(True)
    return uni[keep]

def run_screen(
    today_str: str,
    today_date,
    mkt_score: int,
    delta3: float,
    macro_on: bool,
    state: Dict,
) -> Tuple[List[Dict], Dict, Dict[str, pd.DataFrame]]:
    """
    戻り: (final_candidates_for_line, debug_meta, ohlc_map)
    ※ 歪みEVは内部処理のみ（LINE非表示）
    """
    if not os.path.exists(UNIVERSE_PATH):
        return [], {"raw": 0, "final": 0, "avgAdjEV": 0.0, "GU": 0.0}, {}

    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], {"raw": 0, "final": 0, "avgAdjEV": 0.0, "GU": 0.0}, {}

    tcol = _get_ticker_col(uni)
    if not tcol:
        return [], {"raw": 0, "final": 0, "avgAdjEV": 0.0, "GU": 0.0}, {}

    uni = _filter_earnings(uni, today_date)
    tickers = uni[tcol].astype(str).tolist()
    ohlc_map = download_history_bulk(tickers, period="260d", auto_adjust=True, group_size=200)

    # paper trade update
    update_paper_trades_with_ohlc(state, "tier0_exception", ohlc_map, today_str)
    update_paper_trades_with_ohlc(state, "distortion", ohlc_map, today_str)

    # distortion KPI -> auto OFF
    kpi = kpi_distortion(state)
    if kpi["count"] >= 10:
        if (kpi["median_r"] < -0.10) or (kpi["exp_gap"] < -0.30) or (kpi["neg_streak"] >= 3):
            set_cooldown_days(state, "distortion_until", days=4)

    no_trade = no_trade_conditions(int(mkt_score), float(delta3))

    cands: List[Dict] = []
    gu_cnt = 0

    for _, row in uni.iterrows():
        ticker = str(row.get(tcol, "")).strip()
        if not ticker:
            continue
        df = ohlc_map.get(ticker)
        if df is None or df.empty or len(df) < 120:
            continue

        if is_abnormal_stock(df):
            continue

        ok_liq, price, adv, atrp = liquidity_filters(df)
        if not ok_liq:
            continue

        info = build_setup_info(df, macro_on=macro_on)
        if info.setup == "NONE":
            continue

        if info.gu:
            gu_cnt += 1

        ev, passes, debug = calc_ev(info, mkt_score=int(mkt_score), macro_on=macro_on)
        ok, _ = pass_thresholds(info, ev)
        if not ok:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        cands.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "setup": info.setup,
                "tier": int(info.tier),
                "entry_low": float(info.entry_low),
                "entry_high": float(info.entry_high),
                "sl": float(info.sl),
                "tp1": float(info.tp1),
                "tp2": float(info.tp2),
                "rr": float(ev.rr),
                "struct_ev": float(ev.structural_ev),
                "adj_ev": float(ev.adj_ev),
                "expected_days": float(ev.expected_days),
                "rday": float(ev.rday),
                "gu": bool(info.gu),
                "adv20": float(adv),
                "atrp": float(atrp),
            }
        )

    cands.sort(key=lambda x: (x["adj_ev"], x["rday"], x["rr"]), reverse=True)
    raw_n = len(cands)

    # diversify
    cands = apply_sector_cap(cands, max_per_sector=2)
    cands = apply_corr_filter(cands, ohlc_map, max_corr=0.75)

    final: List[Dict] = []

    if no_trade:
        # Tier0 exception: max 1, cooldownあり
        if not in_cooldown(state, "tier0_exception_until"):
            tier0 = [c for c in cands if c.get("setup") == "A1-Strong"]
            if tier0:
                pick = tier0[0]
                final = [pick]
                entry_mid = (pick["entry_low"] + pick["entry_high"]) / 2.0
                record_paper_trade(
                    state,
                    bucket="tier0_exception",
                    ticker=pick["ticker"],
                    date_str=today_str,
                    entry=entry_mid,
                    sl=pick["sl"],
                    tp2=pick["tp2"],
                    expected_r=float(pick["rr"]),
                )
    else:
        final = cands[:max_display(macro_on)]

    # Tier2 liquidity cushion
    filtered = []
    for c in final:
        if c.get("tier") == 2 and c.get("setup") in ("A2", "B"):
            if float(c.get("adv20", 0.0)) < 300e6:
                continue
        filtered.append(c)
    final = filtered

    # distortion internal (non-display)
    if not in_cooldown(state, "distortion_until"):
        internal = [c for c in cands if c.get("setup") in ("A1-Strong", "A2")][:2]
        for c in internal:
            entry_mid = (c["entry_low"] + c["entry_high"]) / 2.0
            record_paper_trade(
                state,
                bucket="distortion",
                ticker=c["ticker"],
                date_str=today_str,
                entry=entry_mid,
                sl=c["sl"],
                tp2=c["tp2"],
                expected_r=float(c["rr"]),
            )

    # Tier0 exception brake
    pt = state.get("paper_trades", {}).get("tier0_exception", [])
    closed = [x for x in pt if x.get("status") == "CLOSED" and x.get("realized_r") is not None]
    if len(closed) >= 4:
        lastN = closed[-4:]
        s = float(np.sum([safe_float(x.get("realized_r"), 0.0) for x in lastN]))
        if s <= -2.0:
            set_cooldown_days(state, "tier0_exception_until", days=4)

    avg_adj = float(np.mean([c["adj_ev"] for c in final])) if final else 0.0
    gu_ratio = float(gu_cnt / max(1, raw_n)) if raw_n > 0 else 0.0

    meta = {
        "raw": int(raw_n),
        "final": int(len(final)),
        "avgAdjEV": float(avg_adj),
        "GU": float(gu_ratio),
    }

    return final, meta, ohlc_map
