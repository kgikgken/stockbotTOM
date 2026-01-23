from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils.diversify import apply_diversification
from utils.market import market_summary
from utils.position import read_positions
from utils.report import render_daily_report
from utils.rr_ev import (
    rr as rr_calc,
    expected_days as exp_days_calc,
    turnover_efficiency,
    expected_value,
    cagr_contribution,
    adj_ev,
)
from utils.screen_logic import build_raw_candidates
from utils.setup import build_setup_info
from utils.util import atr14


@dataclass
class Config:
    max_display: int = 5
    rr_min: float = 1.8
    adj_ev_min: float = 0.50

    # Setup-specific turnover floors (R/day)
    rday_min_a1: float = 0.45
    rday_min_a1_strong: float = 0.50
    rday_min_breakout: float = 0.65
    rday_min_distortion: float = 0.35

    # Time efficiency penalty (days => multiplicative factor)
    penalty_4d: float = 0.95
    penalty_5d: float = 0.90


def _index_vol_bucket(index_ticker: str = "1306.T") -> str:
    """Return index volatility bucket: low / mid / high.

    Spec: single index vol source (TOPIX ATR14). We approximate with ATR% of 14.
    """
    try:
        df = yf.download(index_ticker, period="6mo", interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty or len(df) < 30:
            return "mid"
        a14 = atr14(df)
        if not np.isfinite(a14):
            return "mid"
        close = float(df["Close"].iloc[-1])
        atr_pct = 100.0 * a14 / max(1e-9, close)
        # Heuristic buckets: TOPIX ETF daily ATR% is generally small.
        if atr_pct <= 1.0:
            return "low"
        if atr_pct >= 2.0:
            return "high"
        return "mid"
    except Exception:
        return "mid"


def _strategy_slot_limits(market_score: int, macro_caution: bool, index_vol_bucket: str) -> Dict[str, int]:
    """Daily auto slot limits per strategy (spec v2.3)."""
    # Pullback = 0..3
    if market_score >= 60:
        pb = 3
    elif market_score >= 45:
        pb = 2
    else:
        pb = 1
    if macro_caution:
        pb = max(0, pb - 1)

    # Breakout = 0..2 (prefer when volatility is rising and market not very strong)
    if index_vol_bucket == "high" and 35 <= market_score <= 70:
        br = 2
    elif 45 <= market_score <= 70:
        br = 1
    else:
        br = 0
    if macro_caution:
        br = max(0, br - 1)

    # Distortion = 0..1 (rare / small)
    ds = 1 if (market_score < 55 or macro_caution) else 0

    return {"A": pb, "B": br, "D": ds}


def _reach_prob(setup: str, row: Dict) -> float:
    """Reach probability proxy (no Pwin)."""
    if setup == "A1-Strong":
        base = 0.62
        q = float(row.get("pullback_score", 0.0)) / 100.0
        return float(np.clip(base * (0.15 + 0.85 * q), 0.35, 0.80))
    if setup == "A1":
        base = 0.55
        q = float(row.get("pullback_score", 0.0)) / 100.0
        return float(np.clip(base * (0.15 + 0.85 * q), 0.30, 0.75))
    if setup == "B":
        base = 0.45
        q = float(row.get("breakout_score", 0.0)) / 100.0
        return float(np.clip(base * (0.20 + 0.80 * q), 0.20, 0.65))
    if setup == "D":
        base = 0.35
        q = float(row.get("distortion_score", 0.0)) / 100.0
        return float(np.clip(base * (0.30 + 0.70 * q), 0.15, 0.55))
    return 0.0


def _rday_floor(setup: str, cfg: Config) -> float:
    if setup == "A1-Strong":
        return cfg.rday_min_a1_strong
    if setup == "A1":
        return cfg.rday_min_a1
    if setup == "B":
        return cfg.rday_min_breakout
    if setup == "D":
        return cfg.rday_min_distortion
    return 9e9


def _time_penalty_factor(exp_days: float, cfg: Config) -> float:
    if not np.isfinite(exp_days):
        return 0.0
    if exp_days >= 6.0:
        return 0.0  # exclude
    if exp_days >= 5.0:
        return cfg.penalty_5d
    if exp_days >= 4.0:
        return cfg.penalty_4d
    return 1.0


def _pick_levels(row: Dict, setup: str) -> Dict:
    if setup in ("A1", "A1-Strong"):
        return row.get("pb_levels", {})
    if setup == "B":
        return row.get("br_levels", {})
    if setup == "D":
        return row.get("ds_levels", {})
    return {}


def _enrich_candidate(
    row: Dict,
    setup: str,
    market_score: int,
    macro_caution: bool,
    risk_on: bool,
    cfg: Config,
) -> Dict | None:
    levels = _pick_levels(row, setup)
    if not levels:
        return None

    entry = float(levels.get("entry", np.nan))
    sl = float(levels.get("sl", np.nan))
    tp1 = float(levels.get("tp1", np.nan))
    tp2 = float(levels.get("tp2", np.nan))
    atr = float(levels.get("atr", np.nan))
    if not (np.isfinite(entry) and np.isfinite(sl) and np.isfinite(tp1) and np.isfinite(atr)):
        return None
    if entry <= sl:
        return None

    rr_tp1 = rr_calc(entry, sl, tp1)
    rr_tp2 = rr_calc(entry, sl, tp2) if np.isfinite(tp2) else float("nan")

    exp_days = exp_days_calc(entry, tp1, atr)
    # clamp to the system horizon
    exp_days = float(np.clip(exp_days, 1.0, 7.0))

    # Time efficiency penalty / exclusion
    time_factor = _time_penalty_factor(exp_days, cfg)
    if time_factor <= 0:
        return None

    p = _reach_prob(setup, row)
    raw_ev = expected_value(rr_tp1, p)
    adj = adj_ev(raw_ev, market_score=market_score, macro_caution=macro_caution, risk_on=risk_on)

    rday = turnover_efficiency(rr_tp1, exp_days)
    cagr = cagr_contribution(rr_tp1, p, exp_days) * time_factor

    # Global filters
    if rr_tp1 < cfg.rr_min:
        return None
    if adj < cfg.adj_ev_min:
        return None
    if rday < _rday_floor(setup, cfg):
        return None

    out = dict(row)
    out.update(
        {
            "setup": setup,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "atr": atr,
            "rr": float(rr_tp1),
            "rr2": float(rr_tp2),
            "reach_prob": float(p),
            "raw_ev": float(raw_ev),
            "adj_ev": float(adj),
            "rday": float(rday),
            "expected_days": float(exp_days),
            "cagr_score": float(cagr),
        }
    )
    return out


def _apply_slot_limits(rows: List[Dict], limits: Dict[str, int]) -> List[Dict]:
    """Enforce daily max slots per strategy."""
    picked: List[Dict] = []
    used = {"A": 0, "B": 0, "D": 0}
    for r in rows:
        s = r.get("setup")
        key = "A" if s in ("A1", "A1-Strong") else s
        if key not in used:
            continue
        if used[key] >= limits.get(key, 0):
            continue
        picked.append(r)
        used[key] += 1
    return picked


def run_screen(universe_csv: str = "data/universe_jpx.csv") -> Tuple[str, Dict]:
    """Run end-to-end screening and return (report_text, debug_dict)."""
    cfg = Config()

    mkt = market_summary()
    market_score = int(mkt["market_score"])
    macro_caution = bool(mkt.get("macro_caution", False))
    risk_on = bool(mkt.get("risk_on", False))

    universe_df = pd.read_csv(universe_csv)
    raw, dbg = build_raw_candidates(universe_df)

    enriched: List[Dict] = []
    for row in raw:
        setup = build_setup_info(row)
        if setup is None:
            continue
        r = _enrich_candidate(row, setup, market_score, macro_caution, risk_on, cfg)
        if r is not None:
            enriched.append(r)

    # Sort by CAGR contribution
    enriched.sort(key=lambda x: float(x.get("cagr_score", -1e9)), reverse=True)

    # Diversification & correlations
    diversified = apply_diversification(enriched)

    # Strategy slot limits
    vol_bucket = _index_vol_bucket(os.getenv("INDEX_TICKER", "1306.T"))
    limits = _strategy_slot_limits(market_score, macro_caution, vol_bucket)
    diversified = _apply_slot_limits(diversified, limits)

    # Cap display
    candidates = diversified[: cfg.max_display]

    positions = read_positions("data/positions.csv")

    report = render_daily_report(mkt, candidates, positions, cfg=cfg, slot_limits=limits)
    dbg.update({"n_raw": len(raw), "n_enriched": len(enriched), "n_final": len(candidates), "slot_limits": limits})
    return report, dbg
