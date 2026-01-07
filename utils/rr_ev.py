# ============================================
# utils/rr_ev.py
# RR / EV / 補正EV / 速度（R/day）計算
# ============================================

from utils.util import safe_div, clamp, rr_min_by_market, estimate_days, calc_r_per_day


# --------------------------------------------
# RR 計算（構造ベース・固定化しない）
# --------------------------------------------
def calc_rr(entry: float, stop: float, tp2: float) -> float:
    """
    RR = (TP2 - Entry) / (Entry - Stop)
    """
    risk = entry - stop
    reward = tp2 - entry
    if risk <= 0:
        return 0.0
    return safe_div(reward, risk)


# --------------------------------------------
# 勝率 proxy（0〜1）
# ※ スコアではなく EV の素材としてのみ使用
# --------------------------------------------
def estimate_pwin(features: dict) -> float:
    """
    features 例:
    - trend_strength (0〜1)
    - rs_strength (0〜1)
    - sector_rank (1〜N)
    - volume_quality (0〜1)
    - gu_flag (bool)
    - liquidity (0〜1)
    """

    p = 0.0

    # トレンドの強さ
    p += 0.30 * features.get("trend_strength", 0.0)

    # 相対強度
    p += 0.25 * features.get("rs_strength", 0.0)

    # セクター順位（上位ほど良い）
    sector_rank = features.get("sector_rank", 10)
    p += 0.15 * clamp(1.0 - (sector_rank - 1) / 10.0, 0.0, 1.0)

    # 出来高の質
    p += 0.20 * features.get("volume_quality", 0.0)

    # 流動性
    p += 0.10 * features.get("liquidity", 0.0)

    # GU は強烈に減点
    if features.get("gu_flag", False):
        p -= 0.30

    return clamp(p, 0.05, 0.80)


# --------------------------------------------
# EV 計算
# --------------------------------------------
def calc_ev(rr: float, pwin: float) -> float:
    """
    EV = Pwin * RR - (1 - Pwin) * 1R
    """
    return pwin * rr - (1.0 - pwin) * 1.0


# --------------------------------------------
# 地合い補正 EV
# --------------------------------------------
def adjust_ev_by_market(
    ev: float,
    market_score: float,
    delta_market: int,
    macro_risk: bool
) -> float:
    """
    地合い・イベントで EV を現実値に補正
    """

    mult = 1.0

    # 地合いレベル
    if market_score >= 70:
        mult *= 1.05
    elif market_score >= 60:
        mult *= 1.00
    elif market_score >= 50:
        mult *= 0.90
    else:
        mult *= 0.80

    # 地合いの変化
    if delta_market <= -5:
        mult *= 0.75
    elif delta_market >= 5:
        mult *= 1.05

    # マクロイベント
    if macro_risk:
        mult *= 0.75

    return ev * mult


# --------------------------------------------
# RR 下限チェック（地合い連動）
# --------------------------------------------
def pass_rr_filter(rr: float, market_score: float) -> bool:
    """
    地合いが悪いほど高RRを要求
    """
    rr_min = rr_min_by_market(market_score)
    return rr >= rr_min


# --------------------------------------------
# EV 足切り
# --------------------------------------------
def pass_ev_filter(adj_ev: float, threshold: float = 0.5) -> bool:
    """
    補正EVが一定未満は不採用
    """
    return adj_ev >= threshold


# --------------------------------------------
# 速度評価（R/day）
# --------------------------------------------
def calc_speed_metrics(
    entry: float,
    tp2: float,
    atr: float,
    rr: float
) -> dict:
    """
    ExpectedDays / R_per_day を算出
    """
    days = estimate_days(tp2, entry, atr)
    r_day = calc_r_per_day(rr, days)

    return {
        "expected_days": days,
        "r_day": r_day,
    }


# --------------------------------------------
# 速度フィルタ
# --------------------------------------------
def pass_speed_filter(
    expected_days: float,
    r_day: float,
    max_days: float = 5.0,
    min_r_day: float = 0.5
) -> bool:
    """
    1〜7日戦用の速度足切り
    """
    if expected_days > max_days:
        return False
    if r_day < min_r_day:
        return False
    return True