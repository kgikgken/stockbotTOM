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
    p_reach_tp1: float
    adj_ev: float
    expected_days: float
    rday: float
    rr_min: float
    rday_min: float
    cagr_contrib: float


def _time_efficiency_penalty(expected_days: float) -> float:
    """時間効率ペナルティ（仕様固定）
    - 1〜3日: 0
    - 4日: -0.05
    - 5日: -0.10
    - 6日以上: 取らない（caller側で除外）
    """
    if expected_days >= 6.0:
        return -999.0
    if expected_days >= 5.0:
        return -0.10
    if expected_days >= 4.0:
        return -0.05
    return 0.0


def calc_ev(setup: SetupInfo, mkt_score: int, macro_on: bool) -> EVInfo:
    """EV計算（表示/順位付けに必要な最小セット）

    - 期待Rは TP1基準で固定（setup.expected_r）
    - CAGR寄与度 = (期待R × 到達確率) ÷ 想定日数（時間効率ペナ込み）
    - MarketScore はゲートにしない（撤退速度制御専用のため、ここでは調整しない）
    """
    rr_min = float(rr_min_by_market(mkt_score))  # 現状は固定1.8として返す想定
    rday_min = float(rday_min_by_setup(setup.setup))

    # 構造EV（きれいな形ほど高い）
    structural_ev = (
        0.45 * float(clamp(setup.trend_strength, 0.0, 1.0))
        + 0.55 * float(clamp(setup.pullback_quality, 0.0, 1.0))
    )
    structural_ev = float(clamp(structural_ev, 0.0, 1.0))

    # TP1到達確率（推定）
    # - 押し目は再現性優先（0.45〜0.75中心）
    # - 初動/歪みは短命・不安定なので上限を抑える
    base = 0.35
    p = base + 0.35 * structural_ev
    if setup.setup in ("A1-Strong",):
        p += 0.05
    if setup.setup in ("A2", "B"):
        p -= 0.05
    if setup.setup in ("S",):
        p -= 0.10
    p = float(clamp(p, 0.15, 0.80))

    # 期待値（補正）: TP1到達を利益、未到達を-1Rとして単純化（スコア安定性優先）
    exp_r = float(setup.expected_r)
    adj_ev = float(exp_r * p - (1.0 - p))

    # 回転効率（期待R/日）
    expected_days = float(setup.expected_days)
    rday = float(exp_r / max(expected_days, 1e-6))

    # CAGR寄与度（時間効率ペナルティ込み）
    pen = _time_efficiency_penalty(expected_days)
    if pen <= -100.0:
        cagr = -999.0
    else:
        cagr = float((exp_r * p) / max(expected_days, 1e-6) + pen)

    return EVInfo(
        rr=float(setup.rr),
        structural_ev=structural_ev,
        p_reach_tp1=p,
        adj_ev=adj_ev,
        expected_days=expected_days,
        rday=rday,
        rr_min=rr_min,
        rday_min=rday_min,
        cagr_contrib=cagr,
    )


def pass_thresholds(setup: SetupInfo, ev: EVInfo) -> Tuple[bool, str]:
    """フィルタ条件。失敗理由を返す。"""
    # 6日以上は原則除外
    if float(setup.expected_days) >= 6.0:
        return False, "想定日数が長い（6日以上）"

    if float(setup.rr) < float(ev.rr_min):
        return False, "RR下限未満"

    # 期待値（補正）下限
    if float(ev.adj_ev) < 0.50:
        return False, "期待値（補正）下限未満"

    # 回転効率（Setup別）
    if float(ev.rday) < float(ev.rday_min):
        return False, "回転効率下限未満"

    return True, ""


def apply_ev_to_setup(setup: SetupInfo, ev: EVInfo) -> SetupInfo:
    """SetupInfoへ計算結果を反映（参照の一貫性維持）。"""
    setup.p_reach_tp1 = float(round(ev.p_reach_tp1, 2))
    setup.adj_ev = float(round(ev.adj_ev, 2))
    setup.cagr_contrib = float(round(ev.cagr_contrib, 3))
    return setup
