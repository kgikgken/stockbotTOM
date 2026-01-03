from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.util import fetch_history, safe_float
from utils.features import compute_features, macro_tag_from_sector
from utils.setup import detect_setup
from utils.entry import build_entry_plan
from utils.rr_ev import compute_rr_ev
from utils.diversify import apply_constraints


# ---- Core thresholds (spec)
LIQ_ADV20_MIN = 200_000_000.0
PRICE_MIN = 200.0
PRICE_MAX = 15_000.0

ATR_PCT_MIN = 1.5
ATR_PCT_MAX = 6.0  # too wild zone: exclude

EARNINGS_EXCLUDE_DAYS = 3  # business-day approx not perfect (calendar fallback)

RR_MIN = 2.0
EV_MIN = 0.3
ADJ_EV_MIN = 0.3

EXPECTED_DAYS_MAX = 5.0
R_PER_DAY_MIN = 0.5

RDAY_CAP_WHEN_WEAK = 0.85  # spec③

MAX_FINAL = 5
WATCH_MAX = 10


def _business_days_diff(d1, d2) -> int:
    # simple: pandas bdate_range (Japan holidays not included)
    try:
        if d1 is None or d2 is None:
            return 999
        a = pd.Timestamp(d1)
        b = pd.Timestamp(d2)
        if a == b:
            return 0
        rng = pd.bdate_range(min(a, b), max(a, b))
        return int(len(rng) - 1)
    except Exception:
        try:
            return abs((d1 - d2).days)
        except Exception:
            return 999


def filter_earnings(universe: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in universe.columns:
        return universe
    df = universe.copy()
    parsed = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    keep = []
    for d in parsed:
        if d is None or pd.isna(d):
            keep.append(True)
            continue
        dd = _business_days_diff(d, today_date)
        keep.append(dd > EARNINGS_EXCLUDE_DAYS)
    return df[keep]


def _hh20_break(hist: pd.DataFrame) -> Tuple[bool, float, bool]:
    close = hist["Close"].astype(float)
    high = hist["High"].astype(float)
    vol = hist["Volume"].astype(float) if "Volume" in hist.columns else None

    if len(close) < 25:
        return False, float("nan"), False

    hh20 = float(high.tail(21).iloc[:-1].max())  # exclude last bar for clean breakout line
    c = float(close.iloc[-1])

    hh_break = bool(np.isfinite(hh20) and np.isfinite(c) and c > hh20)

    vol_ok = False
    if vol is not None and len(vol) >= 25:
        v_last = safe_float(vol.iloc[-1])
        v_ma20 = safe_float(vol.rolling(20).mean().iloc[-1])
        vol_ok = bool(np.isfinite(v_last) and np.isfinite(v_ma20) and v_ma20 > 0 and v_last >= 1.5 * v_ma20)

    return hh_break, float(hh20), bool(vol_ok)


def run_swing_screening(
    today_date,
    universe_path: str,
    market: Dict,
    sector_rank_map: Dict[str, int],
    regime_multiplier: float,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
      - selected core candidates (max 5)
      - watch list (max 10)
    """
    if not os.path.exists(universe_path):
        return [], [{"ticker": "-", "reason": "universe_jpx.csvが見つかりません"}]

    try:
        uni = pd.read_csv(universe_path)
    except Exception:
        return [], [{"ticker": "-", "reason": "universe_jpx.csvが読み込めません"}]

    # ticker column
    if "ticker" in uni.columns:
        t_col = "ticker"
    elif "code" in uni.columns:
        t_col = "code"
    else:
        return [], [{"ticker": "-", "reason": "ticker列がありません"}]

    # sector column
    if "sector" in uni.columns:
        s_col = "sector"
    elif "industry_big" in uni.columns:
        s_col = "industry_big"
    else:
        s_col = None

    mkt_score = int(market.get("score", 50))

    # earnings filter (new entries)
    uni = filter_earnings(uni, today_date)

    raw: List[Dict] = []
    watch: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker)).strip()
        sector = str(row.get(s_col, "不明")).strip() if s_col else "不明"
        macro_tag = macro_tag_from_sector(sector)
        sec_rank = sector_rank_map.get(sector)

        hist = fetch_history(ticker, period="320d")
        if hist is None or len(hist) < 120:
            continue

        fp = compute_features(hist)

        # Universe filters
        if not np.isfinite(fp.close) or fp.close < PRICE_MIN or fp.close > PRICE_MAX:
            continue

        if not np.isfinite(fp.adv20_jpy) or fp.adv20_jpy < LIQ_ADV20_MIN:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "流動性弱(ADV20<200M)"})
            continue

        if not np.isfinite(fp.atr_pct) or fp.atr_pct < ATR_PCT_MIN:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "ボラ不足(ATR%<1.5)"})
            continue
        if fp.atr_pct >= ATR_PCT_MAX:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": "ボラ過大(ATR%>=6)"})
            continue

        hh_break, hh20, vol_break_ok = _hh20_break(hist)
        setup = detect_setup(fp, hh_break, vol_break_ok)
        if not setup.ok:
            continue

        entry = build_entry_plan(fp, setup.setup_type, hh20)

        # GU / 乖離はここで watch 落ち
        if entry.action == "WATCH_ONLY":
            watch.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "sector": sector,
                    "reason": entry.reason,
                    "macro_tag": macro_tag,
                }
            )
            continue

        rr_ev = compute_rr_ev(
            hist=hist,
            fp=fp,
            entry_center=entry.in_center,
            setup_type=setup.setup_type,
            sector_rank=sec_rank,
            regime_multiplier=regime_multiplier,
            rday_cap_when_weak=RDAY_CAP_WHEN_WEAK,
            mkt_score=mkt_score,
        )

        if rr_ev.reason.startswith("R/day過大"):
            watch.append({"ticker": ticker, "name": name, "sector": sector, "reason": rr_ev.reason, "macro_tag": macro_tag})
            continue

        if rr_ev.rr < RR_MIN:
            continue
        if rr_ev.ev < EV_MIN:
            continue
        if rr_ev.adj_ev < ADJ_EV_MIN:
            continue
        if rr_ev.expected_days > EXPECTED_DAYS_MAX:
            continue
        if rr_ev.r_per_day < R_PER_DAY_MIN:
            continue

        # attach return series for correlation
        ret20 = hist["Close"].astype(float).pct_change(fill_method=None).tail(20).reset_index(drop=True)

        raw.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "sector_rank": sec_rank if sec_rank is not None else None,
                "macro_tag": macro_tag,
                "setup": setup.setup_type,
                "in_center": float(entry.in_center),
                "in_low": float(entry.band_low),
                "in_high": float(entry.band_high),
                "price": float(fp.close),
                "atr": float(fp.atr) if np.isfinite(fp.atr) else float("nan"),
                "gu": "Y" if entry.gu_flag else "N",
                "action": entry.action,
                "action_reason": entry.reason,
                "stop": float(rr_ev.stop),
                "tp1": float(rr_ev.tp1),
                "tp2": float(rr_ev.tp2),
                "rr": float(rr_ev.rr),
                "ev": float(rr_ev.ev),
                "adj_ev": float(rr_ev.adj_ev),
                "expected_days": float(rr_ev.expected_days),
                "r_per_day": float(rr_ev.r_per_day),
                "_ret20": ret20,
            }
        )

    # prioritize: AdjEV desc -> R/day desc -> RR desc
    raw.sort(key=lambda x: (x["adj_ev"], x["r_per_day"], x["rr"]), reverse=True)

    selected, rejected = apply_constraints(
        candidates=raw,
        max_final=MAX_FINAL,
        sector_max=2,
        macro_max=2,  # spec④
        corr_limit=0.75,
    )

    # build watch list from rejected + existing watch, prioritize reasons
    for r in rejected:
        watch.append(
            {
                "ticker": r.get("ticker"),
                "name": r.get("name"),
                "sector": r.get("sector"),
                "reason": r.get("reject_reason", "制約"),
                "macro_tag": r.get("macro_tag", "other"),
                "setup": r.get("setup", "-"),
                "rr": r.get("rr", 0.0),
                "r_per_day": r.get("r_per_day", 0.0),
            }
        )

    # limit watch to 10 with best informativeness
    watch = watch[:WATCH_MAX]

    # remove internal
    for c in selected:
        c.pop("_ret20", None)

    return selected, watch