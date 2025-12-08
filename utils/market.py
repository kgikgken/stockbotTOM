from __future__ import annotations

import yfinance as yf
import numpy as np


# ============================================================
# TOPIX + 日経 + マザーズで「実需に近い地合い」を判定
# ============================================================

def fetch_index_change(symbol: str, period: str = "5d") -> float:
    """
    指数の5日騰落（%）を返す
    例: +2.35 % → 2.35
    """
    try:
        df = yf.Ticker(symbol).history(period=period)
        if df is None or df.empty:
            return 0.0
        first = float(df["Close"].iloc[0])
        last = float(df["Close"].iloc[-1])
        if first <= 0:
            return 0.0
        return (last / first - 1.0) * 100.0
    except Exception:
        return 0.0


def calc_market_score() -> dict:
    """
    地合いスコア（0〜100）
    TOPIX、日経、マザーズを合成し、
    “スイング視点”の地合いを1値にする。
    """

    # — 指数の変化率（5日）
    topix = fetch_index_change("^TOPX")    # TOPIX
    nikkei = fetch_index_change("^N225")   # 日経平均
    mothers = fetch_index_change("^MOTHERS")  # マザーズ

    # — 配分（経験則）
    # TOPIX: 現物需給、日本企業の地合い
    # 日経: 先物要因、海外
    # Mothers: 成長系感応度
    raw = (
        topix * 0.45 +
        nikkei * 0.35 +
        mothers * 0.20
    )

    # — スコア化（平均0くらいを50点）
    # 生の変化率 → 50点を中立として変換
    # +5% ≒ 65点 / +10% ≒ 80点
    score = 50 + raw * 3.0
    score = float(np.clip(score, 0, 100))

    # — コメント（心理誘導）
    if score >= 80:
        comment = "強い。押し目＋ブレイク両方可"
    elif score >= 70:
        comment = "強め。押し目中心に攻め"
    elif score >= 60:
        comment = "やや強め。押し目狙い"
    elif score >= 50:
        comment = "普通。押し目が適正"
    elif score >= 40:
        comment = "弱め。サイズ抑える"
    else:
        comment = "弱い。守り優先"

    return {
        "score": int(round(score)),
        "comment": comment,
        "topix_5d": round(topix, 2),
        "nikkei_5d": round(nikkei, 2),
        "mothers_5d": round(mothers, 2),
    }