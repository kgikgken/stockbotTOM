from __future__ import annotations

import numpy as np
import yfinance as yf


def _chg(symbol: str, days: int) -> float:
    # days=5 -> "6d" で両端取る
    period = f"{days+1}d"
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return 0.0
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
    except Exception:
        return 0.0


def calc_market_score() -> dict:
    """
    0-100 の“シンプル地合い”
    - 日経+TOPIXの短期変化を中心に作る
    """
    nk5 = _chg("^N225", 5)
    tp5 = _chg("^TOPX", 5)

    base = 50.0
    base += float(np.clip((nk5 + tp5) / 2.0, -20, 20))

    score = int(np.clip(round(base), 0, 100))

    if score >= 70:
        comment = "強め"
    elif score >= 60:
        comment = "やや強め"
    elif score >= 50:
        comment = "中立"
    elif score >= 40:
        comment = "弱め"
    else:
        comment = "弱い"

    return {"score": score, "comment": comment, "n225_5d": nk5, "topix_5d": tp5}


def enhance_market_score() -> dict:
    """
    calc_market_score + SOX/NVDA を軽く反映 + Δ3d（推定）
    """
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    # SOX
    try:
        chg = _chg("^SOX", 5)
        score += float(np.clip(chg / 2.0, -5.0, 5.0))
        mkt["sox_5d"] = chg
    except Exception:
        pass

    # NVDA
    try:
        chg = _chg("NVDA", 5)
        score += float(np.clip(chg / 3.0, -4.0, 4.0))
        mkt["nvda_5d"] = chg
    except Exception:
        pass

    score = int(np.clip(round(score), 0, 100))
    mkt["score"] = score

    # Δ3d（指数変化から推定）
    nk3 = _chg("^N225", 3)
    tp3 = _chg("^TOPX", 3)
    delta3d = float(np.clip((nk3 + tp3) / 2.0 * 2.0, -25, 25))
    mkt["delta3d"] = delta3d

    return mkt


def recommend_leverage(mkt_score: int, delta3d: float = 0.0) -> tuple[float, str]:
    """
    基本は地合い。delta3d で崩れ初動を弱める。
    """
    if mkt_score >= 70:
        lev = 2.0
        comment = "強気（押し目＋一部ブレイク）"
    elif mkt_score >= 60:
        lev = 1.7
        comment = "やや強気（押し目メイン）"
    elif mkt_score >= 50:
        lev = 1.3
        comment = "中立（厳選・押し目中心）"
    elif mkt_score >= 40:
        lev = 1.1
        comment = "やや守り（新規ロット小さめ）"
    else:
        lev = 1.0
        comment = "守り（新規かなり絞る）"

    if delta3d <= -8 and mkt_score < 60:
        lev = max(1.0, lev - 0.2)
        comment += " / 崩れ初動でレバ抑制"

    return float(round(lev, 1)), comment


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))