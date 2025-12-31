from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.features import compute_features
from utils.setup import detect_setup
from utils.entry import compute_entry_zone
from utils.rr_ev import compute_rr_ev
from utils.diversify import diversify_candidates
from utils.util import clamp


EARNINGS_EXCLUDE_DAYS = 3

# Universe filters
PRICE_MIN = 200
PRICE_MAX = 15000
ADV20_MIN = 200_000_000  # JPY
ATR_PCT_MIN = 1.5
ATR_PCT_MAX = 6.0

# Trade constraints
RR_MIN = 1.8
EV_MIN = 0.30
RPD_MIN = 0.65  # Phase1

# ExpectedDays control (Phase2)
MAX_DAYS_STRONG = 4.0  # market>=60
MAX_DAYS_NEUTRAL = 4.5  # market<60

# Output
MAX_FINAL = 5
MAX_WATCH = 10

# Risk (lot accident)
RISK_PER_TRADE = 0.015
MAX_CORE_POSITIONS = 3


def _fetch_daily(ticker: str, period: str = "300d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty and len(df) >= 120:
                return df
        except Exception:
            time.sleep(0.3)
    return None


def _fetch_index_close() -> Optional[pd.Series]:
    try:
        df = yf.Ticker("^TOPX").history(period="180d", auto_adjust=True)
        if df is None or df.empty:
            df = yf.Ticker("^N225").history(period="180d", auto_adjust=True)
        if df is None or df.empty:
            return None
        return df["Close"].astype(float)
    except Exception:
        return None


def _parse_earnings_date(x) -> Optional[pd.Timestamp]:
    try:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _earnings_blocked(earnings_date, today_date) -> bool:
    if earnings_date is None:
        return False
    try:
        d = earnings_date.date()
        delta = abs((d - today_date).days)
        return delta <= EARNINGS_EXCLUDE_DAYS
    except Exception:
        return False


def _lot_risk_estimate(cands: List[Dict], total_asset: float, lev: float) -> Dict:
    """
    Estimate worst-case total loss if taking up to MAX_CORE_POSITIONS.
    Risk per trade is fixed % of levered capital.
    """
    cap = float(total_asset) * float(lev)
    per_risk = cap * RISK_PER_TRADE
    take_n = min(len(cands), MAX_CORE_POSITIONS)
    total_loss = per_risk * take_n
    ratio = total_loss / max(total_asset, 1.0)

    return {
        "per_trade_risk_yen": float(per_risk),
        "total_loss_yen": float(total_loss),
        "loss_ratio": float(ratio),
    }


def run_screening(
    universe_df: pd.DataFrame,
    today_date,
    market_ctx: Dict,
    total_asset: float,
) -> Dict:
    """
    Returns dict for report:
      - final: list
      - watch: list
      - stats: avg rr/ev/adj_ev/rpd
      - gu_ratio
      - no_trade_final (includes EV/GU rules too)
      - lot_warn
    """
    mkt_score = int(market_ctx["score"])
    lev = float(market_ctx["lev"])

    idx_close = _fetch_index_close()

    candidates: List[Dict] = []
    watch: List[Dict] = []

    for _, row in universe_df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        # earnings exclude for new entries
        e_raw = row.get("earnings_date", None)
        e_ts = _parse_earnings_date(e_raw) if e_raw is not None else None
        if _earnings_blocked(e_ts, today_date):
            # keep as watch only (informative)
            w = {"ticker": ticker, "name": name, "sector": sector, "watch_reason": "決算近接(±3日)"}
            watch.append(w)
            continue

        hist = _fetch_daily(ticker, period="320d")
        if hist is None:
            continue

        feat = compute_features(hist)

        c = feat["close"]
        if not np.isfinite(c) or c < PRICE_MIN or c > PRICE_MAX:
            continue

        adv20 = feat["adv20"]
        atr_pct = feat["atr_pct"]

        if not np.isfinite(adv20) or adv20 < ADV20_MIN:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "watch_reason": "流動性弱(ADV20<200M)"})
            continue

        if not np.isfinite(atr_pct) or atr_pct < ATR_PCT_MIN:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "watch_reason": "ボラ不足(ATR%<1.5)"})
            continue

        if np.isfinite(atr_pct) and atr_pct > ATR_PCT_MAX:
            watch.append({"ticker": ticker, "name": name, "sector": sector, "watch_reason": "事故ゾーン(ATR%>6)"})
            continue

        setup_type, fail_reason = detect_setup(feat)
        if setup_type == "-":
            continue

        entry_zone = compute_entry_zone(setup_type, feat)

        rr_ev = compute_rr_ev(
            hist=hist,
            feat=feat,
            setup_type=setup_type,
            entry_zone=entry_zone,
            market_ctx=market_ctx,
            index_close=idx_close,
        )

        rr = rr_ev["rr"]
        ev = rr_ev["ev"]
        adj_ev = rr_ev["adj_ev"]
        expected_days = rr_ev["expected_days"]
        rpd = rr_ev["r_per_day"]

        # Phase1: hard cuts
        if rr < RR_MIN:
            continue
        if ev < EV_MIN:
            continue
        if rpd < RPD_MIN:
            continue

        # Phase2: ExpectedDays control with regime
        max_days = MAX_DAYS_STRONG if mkt_score >= 60 else MAX_DAYS_NEUTRAL
        if expected_days > max_days:
            continue

        cdict = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "setup": setup_type,
            "entry": rr_ev["entry"],
            "in_low": rr_ev["in_low"],
            "in_high": rr_ev["in_high"],
            "price_now": float(round(feat["close"], 1)),
            "atr": rr_ev["atr"],
            "gu": "Y" if entry_zone["gu_flag"] else "N",
            "stop": rr_ev["stop"],
            "tp1": rr_ev["tp1"],
            "tp2": rr_ev["tp2"],
            "rr": float(rr),
            "expected_days": float(expected_days),
            "r_per_day": float(rpd),
            "pwin": float(rr_ev["pwin"]),
            "ev": float(ev),
            "adj_ev": float(adj_ev),
            "action": entry_zone["action"],
            "dist_atr": float(entry_zone["dist_atr"]),
            "adv20": float(adv20),
            "atr_pct": float(atr_pct),
            "hist_close": hist["Close"].astype(float),
        }

        candidates.append(cdict)

    # Sort by AdjEV first, then R/day, then RR (RR is not a main driver)
    candidates.sort(key=lambda x: (x["adj_ev"], x["r_per_day"], x["rr"]), reverse=True)

    # Diversify (sector cap + corr)
    kept, moved = diversify_candidates(candidates, max_per_sector=2, corr_threshold=0.75)
    watch.extend(moved)

    # Take top
    final = kept[:MAX_FINAL]

    # Compute GU ratio among final
    gu_cnt = sum(1 for c in final if c["gu"] == "Y")
    gu_ratio = (gu_cnt / max(len(final), 1)) if final else 0.0

    # Phase3: NO-TRADE final conditions (mechanical, includes market + avgAdjEV + GU ratio)
    no_trade = (not bool(market_ctx["allow_new"]))
    no_trade_reasons = list(market_ctx.get("no_trade_reasons", []))

    if final:
        avg_adj_ev = float(np.mean([c["adj_ev"] for c in final]))
        if avg_adj_ev < 0.30:
            no_trade = True
            no_trade_reasons.append("平均AdjEV<0.3R")
    else:
        avg_adj_ev = 0.0

    if gu_ratio >= 0.60:
        no_trade = True
        no_trade_reasons.append("GU比率>=60%")

    # lot accident warning
    lot = _lot_risk_estimate(final, total_asset=total_asset, lev=lev)

    # dynamic leverage downshift if too risky
    loss_ratio = lot["loss_ratio"]
    lev_adj = lev
    lev_reason = ""
    if loss_ratio > 0.10:
        no_trade = True
        no_trade_reasons.append("ロット事故(想定最大損失>10%)")
    elif loss_ratio > 0.08:
        lev_adj = max(1.0, lev - 0.3)
        lev_reason = "ロット事故回避でレバ減"
    lot["lev_adj"] = float(lev_adj)
    lot["lev_reason"] = lev_reason

    # Shrink watch list
    watch = watch[:MAX_WATCH]

    # Stats
    def _avg(key: str) -> float:
        if not final:
            return 0.0
        return float(np.mean([c[key] for c in final]))

    stats = {
        "count": int(len(final)),
        "avg_rr": _avg("rr"),
        "avg_ev": _avg("ev"),
        "avg_adj_ev": _avg("adj_ev"),
        "avg_r_per_day": _avg("r_per_day"),
    }

    return {
        "final": final,
        "watch": watch,
        "stats": stats,
        "gu_ratio": float(gu_ratio),
        "no_trade": bool(no_trade),
        "no_trade_reasons": no_trade_reasons,
        "lot": lot,
    }