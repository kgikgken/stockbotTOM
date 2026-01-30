from __future__ import annotations

from dataclasses import dataclass

from utils.util import clamp

@dataclass
class EVResult:
    expected_r: float         # RR at TP1 (expected-R basis)
    p_reach: float            # probability of reaching TP1 before SL within horizon
    cagr_score: float         # (expected_r * p_reach) / expected_days (after penalties)
    structural_ev: float      # keep for compatibility (same as cagr_score)
    adj_ev: float             # keep for compatibility (same as cagr_score)
    rday: float               # expected_r / expected_days
    expected_days: float

def _time_efficiency_penalty(expected_days: float) -> float:
    """Multiplicative penalty from the latest spec.

    1-3 days: no penalty
    4 days:   -5%
    5 days:   -10%
    6+ days:  exclude upstream (score->0)
    """
    d = float(expected_days)
    if d < 3.5:
        return 1.0
    if d < 4.5:
        return 0.95
    if d < 5.5:
        return 0.90
    return 0.0

def _p_reach_tp1(setup: str, rr_tp1: float, expected_days: float, trend_strength: float, pullback_quality: float,
                market_score: float, atr_pct: float, gu: bool) -> float:
    """Calibrated, bounded probability proxy.

    - Baseline by setup (A1-Strong > A1 > A2 > B)
    - Penalize high RR demand and long horizon
    - Mild market regime factor (MarketScore is NOT a gate; it shifts hit-rate)
    - Penalize GU (execution slippage / chase risk)
    """
    base = {
        "A1-Strong": 0.58,
        "A1": 0.53,
        "A2": 0.48,
        "B": 0.44,
    }.get(setup, 0.45)

    # RR demand penalty: harder to hit larger RR in limited time.
    rr_pen = clamp(1.0 - 0.10 * max(0.0, float(rr_tp1) - 2.0), 0.65, 1.05)

    # Horizon penalty: longer days -> lower hit probability.
    d = float(expected_days)
    day_pen = clamp(1.0 - 0.06 * max(0.0, d - 2.5), 0.60, 1.05)

    # Trend/Pullback factors are already bounded ~[0.8,1.2]
    qual = clamp(0.50 * float(trend_strength) + 0.50 * float(pullback_quality), 0.80, 1.20)

    # Market regime: small, symmetric.
    ms = float(market_score)
    if ms >= 75:
        regime = 1.05
    elif ms >= 60:
        regime = 1.02
    elif ms >= 50:
        regime = 1.00
    else:
        regime = 0.94

    # ATR%: too low -> slow; too high -> noisy. Keep mild.
    ap = float(atr_pct)
    if ap < 2.0:
        vol = 0.93
    elif ap < 4.0:
        vol = 1.00
    else:
        vol = 0.97

    gu_pen = 0.92 if gu else 1.00

    p = base * rr_pen * day_pen * qual * regime * vol * gu_pen
    return float(clamp(p, 0.18, 0.78))

def calc_ev(setup_info, market_score: float, atr_pct: float) -> EVResult:
    """Return CAGR contribution score inputs.

    Latest spec:
      - expected_r is RR at TP1 (rr_tp1) and is the ONLY expected-R basis.
      - score = (expected_r * p_reach) / expected_days (time-penalized).
      - time efficiency penalty and 6+ day exclusion.
    """
    expected_days = float(setup_info.expected_days)

    # Exclude slow setups (6+ days) by forcing score to 0 (filtered upstream or by score).
    time_mult = _time_efficiency_penalty(expected_days)
    if time_mult <= 0:
        return EVResult(
            expected_r=float(getattr(setup_info, "rr_tp1", setup_info.rr)),
            p_reach=0.0,
            cagr_score=0.0,
            structural_ev=0.0,
            adj_ev=0.0,
            rday=0.0,
            expected_days=expected_days,
        )

    expected_r = float(getattr(setup_info, "rr_tp1", None) or setup_info.rr)

    p_reach = _p_reach_tp1(
        setup=getattr(setup_info, "setup", "A1"),
        rr_tp1=expected_r,
        expected_days=expected_days,
        trend_strength=float(getattr(setup_info, "trend_strength", 1.0)),
        pullback_quality=float(getattr(setup_info, "pullback_quality", 1.0)),
        market_score=float(market_score),
        atr_pct=float(atr_pct),
        gu=bool(getattr(setup_info, "gu", False)),
    )

    cagr_score = (expected_r * p_reach) / max(1.0, expected_days)
    cagr_score *= time_mult

    rday = expected_r / max(1.0, expected_days)

    return EVResult(
        expected_r=expected_r,
        p_reach=float(p_reach),
        cagr_score=float(cagr_score),
        structural_ev=float(cagr_score),
        adj_ev=float(cagr_score),
        rday=float(rday),
        expected_days=float(expected_days),
    )
def pass_thresholds(
    rr: float,
    ev_adj: float,
    rday: float,
    rr_min: float,
    ev_min: float,
    rday_min: float,
) -> bool:
    """
    Final screening gate.
    Spec-compliant hard filters.
    """
    if rr < rr_min:
        return False
    if ev_adj < ev_min:
        return False
    if rday < rday_min:
        return False
    return True