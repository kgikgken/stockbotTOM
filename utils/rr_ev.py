from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from utils.screen_logic import rr_min_fixed, rotation_min_by_setup
from utils.util import clamp
from utils.setup import SetupInfo

# 期待RはTP1到達時Rで固定。最終スコア（表示順）は CAGR寄与度（pt）。
# CAGR寄与度（pt） = 100 * (期待R(TP1) * 到達確率) / 想定日数 - 時間効率ペナルティ
# ペナルティ：4日=-5pt / 5日=-10pt / 6日以上は除外（別処理）

BASE_REACH_PROB: Dict[str, float] = {
    "A1-Strong": 0.62,
    "A1": 0.56,
    "A2": 0.50,
    "B": 0.46,
    "D": 0.40,
}

def time_penalty_pt(expected_days: float) -> float:
    if expected_days < 3.5:
        return 0.0
    if expected_days < 4.5:
        return 5.0
    if expected_days < 5.5:
        return 10.0
    return 9999.0

@dataclass
class EVInfo:
    # 既存互換（screenerが参照）
    rr: float
    structural_ev: float
    adj_ev: float
    expected_days: float
    rday: float
    rr_min: float
    rday_min: float

    # 拡張（report/sort用）
    expected_r_tp1: float
    reach_prob: float
    exp_value: float
    rotation_eff: float
    cagr_pt: float
    time_penalty_pt: float

def calc_ev(setup: SetupInfo, mkt_score: int, macro_on: bool) -> EVInfo:
    # MarketScoreは撤退制御専用のため、ここではフィルタ値の表示用途のみ（RR下限は固定）。
    rr_min = float(rr_min_fixed())
    rday_min = float(rotation_min_by_setup().get(setup.setup, 0.50))

    # R定義
    entry = float(setup.entry_mid)
    sl = float(setup.sl)
    tp1 = float(setup.tp1)
    tp2 = float(setup.tp2)

    risk = max(1e-9, entry - sl)
    expected_r_tp1 = (tp1 - entry) / risk
    rr_tp2 = (tp2 - entry) / risk

    base_p = BASE_REACH_PROB.get(setup.setup, 0.50)

    # 因子は2つに圧縮（自由度暴走を防ぐ）
    ts = clamp(float(setup.trend_strength), 0.75, 1.25)
    pq = clamp(float(setup.pullback_quality), 0.80, 1.20)
    reach_prob = clamp(base_p * ts * pq, 0.05, 0.90)

    expected_days = max(0.5, float(setup.expected_days))
    exp_value = expected_r_tp1 * reach_prob
    rotation_eff = exp_value / expected_days

    pen = time_penalty_pt(expected_days)
    cagr_pt = 100.0 * rotation_eff - (0.0 if pen >= 9999.0 else pen)

    return EVInfo(
        rr=float(rr_tp2),
        structural_ev=float(exp_value),
        adj_ev=float(exp_value),
        expected_days=float(expected_days),
        rday=float(rotation_eff),
        rr_min=float(rr_min),
        rday_min=float(rday_min),
        expected_r_tp1=float(expected_r_tp1),
        reach_prob=float(reach_prob),
        exp_value=float(exp_value),
        rotation_eff=float(rotation_eff),
        cagr_pt=float(cagr_pt),
        time_penalty_pt=float(pen),
    )
