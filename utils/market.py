from __future__ import annotations

import numpy as np
import yfinance as yf

def _safe_hist(symbol: str, period: str = "6d", interval: str | None = None):
    try:
        t = yf.Ticker(symbol)
        if interval:
            df = t.history(period=period, interval=interval, auto_adjust=True)
        else:
            df = t.history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None

def _five_day_chg(symbol: str) -> float:
    df = _safe_hist(symbol, period="6d")
    if df is None or len(df) < 2:
        return 0.0
    close = df["Close"].astype(float)
    return float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)

def _one_day_chg(symbol: str) -> float:
    df = _safe_hist(symbol, period="3d")
    if df is None or len(df) < 2:
        return 0.0
    close = df["Close"].astype(float)
    # 最終値と直前値（週末などで欠けてもOK）
    return float((close.iloc[-1] / close.iloc[-2] - 1.0) * 100.0)

def _try_futures_change() -> tuple[float, str]:
    """
    先物リスクオン判定用（夜間の勢いを拾う）
    - Yahoo Finance上の銘柄は時期で変わり得るため、複数シンボルをフォールバック
    戻り: (pct, symbol_used)
    """
    candidates = [
        "NKD=F",   # Nikkei/USD futures (CME)
        "NIY=F",   # Nikkei/Yen futures (CME)
        "NK-F26.SI",  # SGX current-ish contract (example)
    ]
    for sym in candidates:
        pct = _one_day_chg(sym)
        if np.isfinite(pct) and abs(pct) > 0:
            return float(pct * 100.0), sym  # convert to %*100? wait pct already fraction? _one_day_chg returns percent
    return 0.0, ""

def calc_market_score() -> dict:
    """
    日経平均・TOPIXの5日変化で 0-100 の地合いスコア（ベース）
    """
    nk = _five_day_chg("^N225")
    tp = _five_day_chg("^TOPX")

    base = 50.0
    base += float(np.clip((nk + tp) / 2.0, -20, 20))

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

    return {"score": score, "comment": comment, "n225_5d": nk, "topix_5d": tp}

def enhance_market_score() -> dict:
    """
    calc_market_score + SOX/NVDA + 先物（Risk-ON）
    """
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    # SOX
    sox = _safe_hist("^SOX", period="6d")
    if sox is not None and len(sox) >= 2:
        chg = float((sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1.0) * 100.0)
        score += float(np.clip(chg / 2.0, -5.0, 5.0))
        mkt["sox_5d"] = chg

    # NVDA
    nv = _safe_hist("NVDA", period="6d")
    if nv is not None and len(nv) >= 2:
        chg = float((nv["Close"].iloc[-1] / nv["Close"].iloc[0] - 1.0) * 100.0)
        score += float(np.clip(chg / 3.0, -4.0, 4.0))
        mkt["nvda_5d"] = chg

    # Futures Risk-ON (override macro cap)
    fut_pct = 0.0
    fut_sym = ""
    # Use percent change (already %) from _one_day_chg
    for sym in ["NKD=F", "NIY=F", "NK-F26.SI"]:
        fut_pct = _one_day_chg(sym)  # %
        if np.isfinite(fut_pct) and abs(fut_pct) > 0:
            fut_sym = sym
            break
    mkt["futures_1d"] = float(fut_pct)
    mkt["futures_symbol"] = fut_sym

    score = int(np.clip(round(score), 0, 100))
    mkt["score"] = score

    # Risk-ON 判定（仕様）
    # - 先物 +1.0%以上
    # - または MarketScore≥65 & Δ3d≥+3（Δ3dはstate側で算出）
    mkt["futures_risk_on"] = bool(np.isfinite(fut_pct) and fut_pct >= 1.0)

    return mkt
