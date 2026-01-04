from __future__ import annotations

import numpy as np
import yfinance as yf


def _history(symbol: str, period: str = "200d"):
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _ret(df, n: int) -> float:
    if df is None or len(df) < n + 1:
        return 0.0
    c = df["Close"].astype(float)
    return float((c.iloc[-1] / c.iloc[-(n + 1)] - 1.0) * 100.0)


def _ma(df, w: int) -> float:
    if df is None or len(df) < w:
        return np.nan
    return float(df["Close"].astype(float).rolling(w).mean().iloc[-1])


def calc_market_score() -> dict:
    """
    0-100：指数のトレンド + モメンタム
    """
    n225 = _history("^N225", "260d")
    topx = _history("^TOPX", "260d")

    r5 = (_ret(n225, 5) + _ret(topx, 5)) / 2.0
    r20 = (_ret(n225, 20) + _ret(topx, 20)) / 2.0

    # Trend check
    c_n = float(n225["Close"].iloc[-1]) if n225 is not None else 0.0
    ma50_n = _ma(n225, 50)
    ma20_n = _ma(n225, 20)

    trend = 0.0
    if np.isfinite(c_n) and np.isfinite(ma20_n) and np.isfinite(ma50_n):
        if c_n > ma50_n and ma20_n > ma50_n:
            trend = 1.0
        elif c_n > ma50_n:
            trend = 0.6
        else:
            trend = 0.2

    # Score mapping
    base = 50.0
    base += np.clip(r5, -10, 10) * 1.2
    base += np.clip(r20, -15, 15) * 0.8
    base += (trend - 0.5) * 20.0

    score = int(np.clip(round(base), 0, 100))
    comment = (
        "強め" if score >= 70 else
        "やや強め" if score >= 60 else
        "中立" if score >= 50 else
        "弱め" if score >= 40 else
        "弱い"
    )

    return {"score": score, "comment": comment, "r5": r5, "r20": r20}


def enhance_market_score() -> dict:
    """
    calc_market_score + SOX/NVDAを軽く反映（過剰に振らない）
    """
    m = calc_market_score()
    score = float(m["score"])

    # SOX
    sox = _history("^SOX", "60d")
    sox5 = _ret(sox, 5)
    score += float(np.clip(sox5 / 2.0, -5.0, 5.0))
    m["sox_5d"] = sox5

    # NVDA
    nv = _history("NVDA", "60d")
    nv5 = _ret(nv, 5)
    score += float(np.clip(nv5 / 3.0, -4.0, 4.0))
    m["nvda_5d"] = nv5

    m["score"] = int(np.clip(round(score), 0, 100))
    # comment再計算
    sc = m["score"]
    m["comment"] = (
        "強め" if sc >= 70 else
        "やや強め" if sc >= 60 else
        "中立" if sc >= 50 else
        "弱め" if sc >= 40 else
        "弱い"
    )
    return m


def calc_delta_market_score_3d() -> int:
    """
    ΔMarketScore_3d：直近3営業日前との差分（指数ベースの簡易）
    """
    # 直近値
    now = calc_market_score()["score"]
    # 3日前近似：N225/TOPXの3日前で再計算（厳密でなくてOK。用途は安全側判定）
    n225 = _history("^N225", "30d")
    topx = _history("^TOPX", "30d")
    if n225 is None or topx is None or len(n225) < 6 or len(topx) < 6:
        return 0

    # “3日前の終値”を使ったリターン近似でスコア差を再構成
    # → 指標は同系統なので差分の方向性が出ればOK
    def score_at(idx_back: int) -> int:
        c_n = float(n225["Close"].iloc[-1 - idx_back])
        c_t = float(topx["Close"].iloc[-1 - idx_back])
        # 5日/20日も同じidx_backで近似
        r5 = 0.0
        r20 = 0.0
        if len(n225) > 6 + idx_back and len(topx) > 6 + idx_back:
            r5 = ((c_n / float(n225["Close"].iloc[-6 - idx_back]) - 1.0) * 100.0 +
                  (c_t / float(topx["Close"].iloc[-6 - idx_back]) - 1.0) * 100.0) / 2.0
        if len(n225) > 21 + idx_back and len(topx) > 21 + idx_back:
            r20 = ((c_n / float(n225["Close"].iloc[-21 - idx_back]) - 1.0) * 100.0 +
                   (c_t / float(topx["Close"].iloc[-21 - idx_back]) - 1.0) * 100.0) / 2.0
        base = 50.0 + np.clip(r5, -10, 10) * 1.2 + np.clip(r20, -15, 15) * 0.8
        return int(np.clip(round(base), 0, 100))

    past = score_at(idx_back=3)
    return int(now - past)


def market_regime_multiplier(mkt_score: int, delta3d: int, macro_danger: bool) -> float:
    """
    AdjEV補正：地合い×変化速度×イベント
    """
    mul = 1.0
    if mkt_score >= 60 and delta3d >= 0:
        mul *= 1.05
    if delta3d <= -5:
        mul *= 0.70
    if macro_danger:
        mul *= 0.75
    return float(mul)