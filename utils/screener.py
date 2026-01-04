from __future__ import annotations

import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict

from utils.features import compute_indicators, setup_type
from utils.setup import universe_filter, gu_flag
from utils.entry import entry_zone
from utils.rr_ev import rr_from_structure, pwin_proxy, calc_ev
from utils.sector import sector_rank_map
from utils.diversify import apply_diversify


EARNINGS_EXCLUDE_DAYS = 3

RR_MIN = 1.8
EV_MIN = 0.30
RDAY_MIN = 0.50
EXPECTED_DAYS_MAX = 5.0


def _read_universe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _ticker_col(df: pd.DataFrame) -> str | None:
    if "ticker" in df.columns:
        return "ticker"
    if "code" in df.columns:
        return "code"
    return None


def _filter_earnings(df: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df
    d = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    out = df.copy()
    out["_ed"] = d
    keep = []
    for x in out["_ed"]:
        if x is None or pd.isna(x):
            keep.append(True)
            continue
        try:
            keep.append(abs((x - today_date).days) > EARNINGS_EXCLUDE_DAYS)
        except Exception:
            keep.append(True)
    return out.loc[keep].drop(columns=["_ed"], errors="ignore")


def run_swing_screening(
    today_date,
    universe_path: str,
    mkt_score: int,
    delta3d: int,
    macro_danger: bool,
    regime_mul: float,
    weekly_remain: int,
) -> Dict:
    """
    仕様の中枢。
    - Universe足切り
    - A/B setup
    - IN帯/乖離/行動
    - GU禁止（監視に落とす）
    - RR（自然分布）/EV/AdjEV/R/day
    - 速度足切り
    - 分散（セクター上限/相関）
    - NO-TRADE判定用に平均AdjEV等も返す
    """
    uni = _read_universe(universe_path)
    tcol = _ticker_col(uni)
    if tcol is None or uni.empty:
        return {
            "no_trade_hint": True,
            "no_trade_reason_hint": "universe読込失敗",
            "picked": [],
            "watch": [],
            "stats": {},
        }

    # 決算回避（完全）
    uni = _filter_earnings(uni, today_date)

    # セクター順位（補助）
    sec_rank = sector_rank_map(universe_path)

    cands: List[Dict] = []
    watch_base: List[Dict] = []

    # 週次枠ゼロなら候補自体は出すが、行動は監視に寄せる
    weekly_lock = weekly_remain <= 0

    for _, row in uni.iterrows():
        ticker = str(row.get(tcol, "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        try:
            hist = yf.Ticker(ticker).history(period="320d", auto_adjust=True)
        except Exception:
            continue
        if hist is None or len(hist) < 120:
            continue

        ind = compute_indicators(hist)
        ind["sector"] = sector

        ok, reason = universe_filter(ind)
        if not ok:
            watch_base.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "drop_reason": reason,
            })
            continue

        setup = setup_type(ind)
        ez = entry_zone(setup, ind)
        gu = gu_flag(hist)

        rrinfo = rr_from_structure(hist, setup, ez["in_center"], mkt_score)
        rr = float(rrinfo["rr"])
        if rr < RR_MIN:
            watch_base.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "drop_reason": f"RR不足(RR<{RR_MIN})",
            })
            continue

        # pwin proxy -> EV
        sr = int(sec_rank.get(sector, 999))
        pwin = pwin_proxy(setup, sr, float(ind["adv20"]), float(ind["rsi14"]), float(ez["deviation"]))
        ev = calc_ev(rr, pwin)
        if ev < EV_MIN:
            watch_base.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "drop_reason": f"EV不足(EV<{EV_MIN})",
            })
            continue

        # 速度
        expd = float(rrinfo["expected_days"])
        rday = rr / expd if expd > 0 else 0.0
        if expd > EXPECTED_DAYS_MAX or rday < RDAY_MIN:
            watch_base.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "drop_reason": "速度/効率",
            })
            continue

        adj_ev = ev * float(regime_mul)

        action = ez["action"]
        if gu:
            action = "監視のみ"
        if macro_danger:
            action = "監視のみ"
        if weekly_lock:
            action = "監視のみ"

        cands.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "setup": setup,
            "sector_rank": sr,
            "in_center": float(ez["in_center"]),
            "in_low": float(ez["in_low"]),
            "in_high": float(ez["in_high"]),
            "price_now": float(ind["close"]),
            "atr": float(rrinfo["atr"]),
            "gu": bool(gu),
            "stop": float(rrinfo["stop"]),
            "tp1": float(rrinfo["tp1"]),
            "tp2": float(rrinfo["tp2"]),
            "rr": float(rr),
            "pwin": float(pwin),
            "ev": float(ev),
            "adj_ev": float(adj_ev),
            "expected_days": float(expd),
            "r_per_day": float(rday),
            "action": action,
        })

    # 並び順：R/day主導 → AdjEV → RR
    cands.sort(key=lambda x: (x["r_per_day"], x["adj_ev"], x["rr"]), reverse=True)

    picked, watch_div = apply_diversify(cands, sector_max=2, corr_limit=0.75)

    # watchは “落ち理由” を付けて最大10
    watch = []
    watch.extend(watch_div)
    watch.extend(watch_base)
    watch = watch[:10]

    # no-trade hint
    stats = {}
    if picked:
        stats["avg_rr"] = float(np.mean([x["rr"] for x in picked]))
        stats["avg_ev"] = float(np.mean([x["ev"] for x in picked]))
        stats["avg_adj_ev"] = float(np.mean([x["adj_ev"] for x in picked]))
        stats["avg_rday"] = float(np.mean([x["r_per_day"] for x in picked]))
    else:
        stats["avg_rr"] = 0.0
        stats["avg_ev"] = 0.0
        stats["avg_adj_ev"] = 0.0
        stats["avg_rday"] = 0.0

    return {
        "picked": picked,
        "watch": watch,
        "stats": stats,
    }