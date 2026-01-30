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
    """TP1到達確率の簡易モデル。

    目的：CAGR寄与度の分子（期待R×到達確率）に分散を作り、
    "良い形"の中で優劣が付くようにする。

    - 形の質（trend_strength / pullback_quality）
    - RRの難易度（高RRほど到達しにくい）
    - 想定日数（長いほど不確実性↑）
    - 地合い（MarketScore）は弱めに補正
    """

    ts = float(setup.trend_strength)
    pq = float(setup.pullback_quality)
    rr = float(setup.rr)
    days = float(max(setup.expected_days, 0.5))

    base = 0.55
    base += 0.12 * (ts - 1.0)
    base += 0.10 * (pq - 1.0)

    if setup.setup == "A1-Strong":
        base += 0.04
    elif setup.setup == "A1":
        base += 0.02
    elif setup.setup == "A2":
        base -= 0.02
    elif setup.setup == "B":
        base -= 0.06

    # RRが高いほど到達確率は下がる（難易度ペナルティ）
    base -= 0.06 * max(0.0, rr - 2.0)

    # 日数が長いほど不確実性が増える
    base -= 0.03 * max(0.0, days - 3.0)

    p = float(clamp(base, 0.18, 0.85))

    # 地合い補正（過剰反応は避ける）
    if mkt_score is not None:
        try:
            ms = float(mkt_score)
            # 50を中立として±0.08の範囲で線形補正
            adj = clamp((ms - 50.0) / 100.0, -0.08, 0.08)
            p = float(clamp(p + adj, 0.15, 0.88))
        except Exception:
            pass

    return p


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
