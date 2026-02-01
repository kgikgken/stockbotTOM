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


def _reach_prob(setup: SetupInfo, mkt_score: int | None = None) -> float:
    """TP1到達確率（目安）

    仕様（監査OS）：
      - setup別ベース率で初期分散を確保
      - trend_strength / pullback_quality を弱く反映
      - RR(TP1) / 想定日数 を反映
      - ATR%（ボラ）/ ADV20（流動性）で微調整
      - MarketScore は環境補正として弱くのみ反映（撤退速度専用思想を尊重）
    """
    rr_tp1 = float(setup.rr_tp1) if setup.rr_tp1 is not None else float(setup.rr)
    days = max(0.5, float(setup.expected_days))

    base_map = {
        "A1-Strong": 0.62,
        "A1": 0.58,
        "A2": 0.55,
        "B": 0.50,
    }
    p = base_map.get(setup.setup, 0.52)

    q = clamp(float(setup.trend_strength) * float(setup.pullback_quality), 0.80, 1.20)
    p += 0.12 * (q - 1.0)

    p += 0.10 * clamp((rr_tp1 - 1.8) / 1.6, -0.3, 0.9)
    p -= 0.10 * clamp((days - 3.0) / 3.5, 0.0, 0.7)

    atrp = float(setup.atrp) if setup.atrp is not None and np.isfinite(setup.atrp) else np.nan
    if np.isfinite(atrp):
        p -= 0.05 * clamp((atrp - 3.5) / 4.0, -1.0, 1.0)

    adv20 = float(setup.adv20) if setup.adv20 is not None and np.isfinite(setup.adv20) else np.nan
    if np.isfinite(adv20):
        p += 0.04 * clamp((np.log10(max(adv20, 1.0)) - 8.6) / 0.9, -1.0, 1.0)

    if mkt_score is not None:
        env = clamp((int(mkt_score) - 55) / 25.0, -1.0, 1.0)
        p += 0.03 * env

    return float(clamp(p, 0.30, 0.85))

def calc_ev(setup: SetupInfo, mkt_score: int, macro_on: bool) -> EVInfo:
    """CAGR寄与度一本化（TP1基準）。

    - 期待RはTP1基準で固定
    - CAGR寄与度 = (期待R × 到達確率) ÷ 想定日数
    - 時間効率ペナルティ（完全機械）を減点として反映
    - MarketScoreは撤退速度制御専用（選別ゲートにしない）
      → 本関数ではスコアに直接掛けない
    """
    rr_min = float(rr_min_by_market(mkt_score))
    rday_min = float(rday_min_by_setup(setup.setup))

    # Latest spec: RRはTP1基準（=期待Rの基準）。TP2は参考表示のみ。
    expected_r = float(setup.rr)
    rr = expected_r
    expected_days = float(max(setup.expected_days, 0.5))

    p = _reach_prob(setup, mkt_score=mkt_score)

    # 時間効率ペナルティ（機械）
    penalty = 0.0
    if expected_days >= 6.0:
        # 原則除外
        return EVInfo(
            rr=rr,
            structural_ev=-999.0,
            adj_ev=-999.0,
            expected_days=expected_days,
            rday=-999.0,
            rr_min=rr_min,
            rday_min=rday_min,
            cagr_score=-999.0,
            expected_r=expected_r,
            p_reach=p,
            time_penalty_pts=99.0,
        )
    if expected_days >= 5.0:
        penalty = 10.0
    elif expected_days >= 4.0:
        penalty = 5.0

    adj_ev = float(expected_r * p)  # 期待値（補正）
    if setup.gu:
        adj_ev -= 0.10
    if macro_on:
        adj_ev -= 0.08

    adj_ev = float(clamp(adj_ev, -0.50, 2.50))
    rday = float(adj_ev / max(expected_days, 1e-6))
    cagr_score = float(rday - (penalty / 100.0))

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
