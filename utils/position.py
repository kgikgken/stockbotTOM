# ============================================
# utils/position.py
# ポジション管理・リスク制御
# ============================================

import csv
import os
from typing import List, Dict

from utils.util import safe_div


# --------------------------------------------
# 設定
# --------------------------------------------
POSITIONS_PATH = "positions.csv"

RISK_PER_TRADE = 0.015        # 1トレードあたり資産リスク（1.5%）
MAX_WEEKLY_NEW = 3            # 週次新規制限
MAX_TOTAL_RISK = 0.10         # 想定最大損失の上限（資産比10%）


# --------------------------------------------
# ポジション読み込み
# --------------------------------------------
def load_positions() -> List[Dict]:
    """
    positions.csv を読み込む
    必須カラム例:
        ticker, entry, stop, size, is_new
    """
    if not os.path.exists(POSITIONS_PATH):
        return []

    positions = []
    with open(POSITIONS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append(row)
    return positions


# --------------------------------------------
# 週次新規カウント
# --------------------------------------------
def count_weekly_new_positions(positions: List[Dict]) -> int:
    """
    is_new == '1' の件数をカウント
    """
    count = 0
    for p in positions:
        if str(p.get("is_new", "0")) == "1":
            count += 1
    return count


# --------------------------------------------
# 想定損失計算
# --------------------------------------------
def calc_position_risk(entry: float, stop: float, size: float) -> float:
    """
    1ポジションあたりの想定損失
    """
    return abs(entry - stop) * size


def calc_total_risk(positions: List[Dict]) -> float:
    """
    全ポジションの想定最大損失合計
    """
    total = 0.0
    for p in positions:
        try:
            entry = float(p["entry"])
            stop = float(p["stop"])
            size = float(p["size"])
            total += calc_position_risk(entry, stop, size)
        except Exception:
            continue
    return total


# --------------------------------------------
# ロット事故チェック
# --------------------------------------------
def check_risk_warning(
    positions: List[Dict],
    account_size: float
) -> Dict:
    """
    ロット事故警告判定
    """
    total_risk = calc_total_risk(positions)
    risk_ratio = safe_div(total_risk, account_size)

    warning = risk_ratio >= MAX_TOTAL_RISK

    return {
        "warning": warning,
        "total_risk": round(total_risk, 0),
        "risk_ratio": round(risk_ratio * 100, 2),
    }


# --------------------------------------------
# 新規可否（ポジション制約）
# --------------------------------------------
def can_open_new_position(
    positions: List[Dict],
    account_size: float
) -> Dict:
    """
    ポジション制約による新規可否
    """
    weekly_new = count_weekly_new_positions(positions)
    risk_info = check_risk_warning(positions, account_size)

    reasons = []

    if weekly_new >= MAX_WEEKLY_NEW:
        reasons.append("週次新規上限")

    if risk_info["warning"]:
        reasons.append("ロット事故リスク")

    return {
        "can_open": len(reasons) == 0,
        "weekly_new": weekly_new,
        "reasons": reasons,
        "risk_info": risk_info,
    }


# --------------------------------------------
# 推奨ロット計算
# --------------------------------------------
def calc_position_size(
    account_size: float,
    entry: float,
    stop: float,
    leverage: float
) -> float:
    """
    資産・レバ・許容リスクから株数を算出
    """
    risk_amount = account_size * RISK_PER_TRADE
    per_share_risk = abs(entry - stop)

    if per_share_risk <= 0:
        return 0.0

    size = safe_div(risk_amount, per_share_risk)
    return round(size * leverage, 2)