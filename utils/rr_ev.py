from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from utils.screen_logic import rr_min_by_market, rday_min_by_setup
from utils.util import clamp
from utils.setup import SetupInfo


@dataclass
class EVInfo:
    rr: float
    structural_ev: float
    adj_ev: float
    expected_days: float
    rday: float
    rr_min: float
    rday_min: float

    # 追加（最新仕様）
    cagr_score: float
    expected_r: float
    p_reach: float
    time_penalty_pts: float


def _reach_prob(setup: SetupInfo) -> float:
    base = 0.35
    base += 0.20 * float(setup.trend_strength)
    base += 0.20 * float(setup.pullback_quality)

    if setup.setup == "A1-Strong":
        base += 0.06
    elif setup.setup == "A1":
        base += 0.03
    elif setup.setup == "B":
        base -= 0.05
    elif setup.setup == "D":
        base -= 0.10

    return float(clamp(base, 0.20, 0.75))


def calc_ev(setup: SetupInfo, rr_min: float, macro_warn: bool, market_score: float) -> EVInfo:
    """Compute expected value metrics used for filtering and ranking.

    Latest spec alignment:
      - RR uses TP1 basis (expectedR).
      - 期待値（補正） = 期待R(TP1) × 到達確率 × 各種補正（Macro/GU/環境）
      - 回転効率（R/日） = 期待R(TP1) ÷ 想定日数（確率は掛けない）
      - CAGR寄与度 = (期待R(TP1) × 到達確率) ÷ 想定日数 に時間効率ペナルティを適用
    """
    expected_r = safe_float(setup.rr_tp1, np.nan)
    if not np.isfinite(expected_r) or expected_r <= 0:
        expected_r = safe_float(setup.rr, 0.0)

    days = max(0.5, safe_float(setup.expected_days, 3.0))

    # Reach probability proxy (bounded). Keep it simple and stable.
    base = 0.35
    p_reach = base + 0.30 * clamp(setup.trend_strength, 0.0, 1.0) + 0.25 * clamp(setup.pullback_quality, 0.0, 1.0)
    if setup.setup.endswith("Strong"):
        p_reach += 0.05
    if setup.setup.startswith("B"):
        p_reach -= 0.05  # breakouts are faster but lower hit-rate
    if setup.gu:
        p_reach -= 0.05
    p_reach = float(clamp(p_reach, 0.20, 0.85))

    # MarketScore is NOT a gate: only a mild multiplier for "補正" (risk-off reduces).
    m_mult = 1.0
    if market_score < 45:
        m_mult = 0.85
    elif market_score < 55:
        m_mult = 0.93
    elif market_score > 70:
        m_mult = 1.03

    # Macro / GU penalty (display-side only)
    macro_mult = 0.92 if macro_warn else 1.0
    gu_mult = 0.95 if setup.gu else 1.0

    # Raw EV components
    adj_ev = float(expected_r * p_reach * m_mult * macro_mult * gu_mult)
    rday = float(expected_r / days)

    # Time-efficiency penalty (mechanical)
    penalty_pct = 0.0
    if days >= 6.0:
        penalty_pct = 1.0  # exclude upstream; keep score 0
    elif days >= 5.0:
        penalty_pct = 0.10
    elif days >= 4.0:
        penalty_pct = 0.05

    cagr_contrib = float((expected_r * p_reach) / days) if penalty_pct < 1.0 else 0.0
    cagr_score = float(cagr_contrib * (1.0 - penalty_pct))

    rr = float(expected_r)

    no_trade = False
    no_trade_reason = None
    # Basic "minimum RR" filter (environment-adjusted upstream)
    if rr < rr_min:
        no_trade = True
        no_trade_reason = "RR下限未満"

    return EVInfo(
        rr=rr,
        adj_ev=adj_ev,
        rday=rday,
        expected_days=float(days),
        p_reach=p_reach,
        cagr_score=cagr_score,
        no_trade=no_trade,
        no_trade_reason=no_trade_reason,
    )

    # structural_ev は監視/ログ用（TP1基準に寄せる）
    structural_ev = float(expected_r)

    return EVInfo(
        rr=rr,
        structural_ev=structural_ev,
        adj_ev=adj_ev,
        expected_days=expected_days,
        rday=rday,
        rr_min=rr_min,
        rday_min=rday_min,
        cagr_score=cagr_score,
        expected_r=expected_r,
        p_reach=p,
        time_penalty_pts=penalty,
    )


def pass_thresholds(setup: SetupInfo, ev: EVInfo) -> Tuple[bool, str]:
    if ev.rr < ev.rr_min:
        return False, "RR"
    if ev.rday < ev.rday_min:
        return False, "RDAY"
    if ev.adj_ev < 0.50:
        return False, "ADJEV"
    return True, "OK"
