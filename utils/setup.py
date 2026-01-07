# ============================================
# utils/setup.py
# チャート形状判定（Setup A1 / A2）
# ============================================

from __future__ import annotations

import pandas as pd
from typing import Dict


# --------------------------------------------
# Setup 判定メイン
# --------------------------------------------
def judge_setup(
    df: pd.DataFrame,
) -> Dict[str, bool | str]:
    """
    チャート形状から Setup を判定する
    戻り値:
      {
        "valid": bool,
        "type": "A1" | "A2" | "",
        "reason": str
      }
    """

    # 必須データ
    for col in ["close", "ma20", "ma50", "rsi"]:
        if col not in df.columns:
            return {
                "valid": False,
                "type": "",
                "reason": f"missing:{col}",
            }

    latest = df.iloc[-1]

    close = latest["close"]
    ma20 = latest["ma20"]
    ma50 = latest["ma50"]
    rsi = latest["rsi"]

    # ----------------------------------------
    # 共通前提（順張り・押し目）
    # ----------------------------------------
    if not (close > ma20 > ma50):
        return {
            "valid": False,
            "type": "",
            "reason": "MA構造不一致",
        }

    if not (35 <= rsi <= 65):
        return {
            "valid": False,
            "type": "",
            "reason": "RSI過熱/弱すぎ",
        }

    # MA20 傾き
    if len(df) < 6:
        return {
            "valid": False,
            "type": "",
            "reason": "データ不足",
        }

    ma20_slope = df["ma20"].iloc[-1] - df["ma20"].iloc[-6]

    if ma20_slope <= 0:
        return {
            "valid": False,
            "type": "",
            "reason": "MA20横ばい/下向き",
        }

    # ----------------------------------------
    # Setup A1 / A2 分離
    # ----------------------------------------

    # A1: 強トレンド・浅い押し
    if close >= ma20:
        return {
            "valid": True,
            "type": "A1",
            "reason": "強トレンド浅押し",
        }

    # A2: やや深い押し（MA20〜MA50）
    if ma50 <= close < ma20:
        return {
            "valid": True,
            "type": "A2",
            "reason": "押し目待ち型",
        }

    return {
        "valid": False,
        "type": "",
        "reason": "押し位置不適合",
    }