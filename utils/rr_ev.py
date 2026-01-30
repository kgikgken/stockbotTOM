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


def _reach_prob(setup: SetupInfo, mkt_score: int) -> float:
    """TP1到達確率（機械推定）。

    仕様思想：
      - ここは「精度改善フェーズ」の余地だが、裁量を入れないため固定関数で推定する。
      - 入力は SetupInfo に含まれる“再現性因子”と、日次環境（MarketScore）だけ。
      - MarketScoreはゲートにしないが、確率（=期待値補正）には弱く効かせてよい。
    """
    # 係数は過度に鋭くしない（個人運用での頑健性優先）
    ts = float(setup.trend_strength)
    pq = float(setup.pullback_quality)
    ed = float(max(setup.expected_days, 0.5))

    # ベース（A1系が主力なので基準はやや高めに置く）
    x = -0.10
    x += 0.85 * (ts - 1.0)
    x += 0.75 * (pq - 1.0)

    # 想定日数が長いほど成功率を下げる（時間=敵）
    x += -0.18 * (ed - 2.5)

    # セットアップ補正（需給例外は低頻度・低上限）
    if setup.setup == "A1-Strong":
        x += 0.18
    elif setup.setup == "A1":
        x += 0.10
    elif setup.setup == "A2":
        x += 0.02
    elif setup.setup == "B":
        x += -0.06
    elif setup.setup == "D":
        x += -0.18

    # GUは追いかけ禁止なので、到達確率を下げて表示優先度を落とす
    if bool(setup.gu):
        x += -0.12

    # MarketScore（撤退制御専用だが“補正期待値”には弱く反映）
    x += 0.06 * ((float(mkt_score) - 60.0) / 10.0)

    # logistic
    p = 1.0 / (1.0 + pow(2.718281828, -x))
    return float(clamp(p, 0.18, 0.82))

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

    p = _reach_prob(setup)

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
