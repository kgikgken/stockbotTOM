from __future__ import annotations

import numpy as np
import yfinance as yf

from utils.risk import AbnormalFlag


def _five_day_chg(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="6d", auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return 0.0
        close = df["Close"].astype(float)
        return float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
    except Exception:
        return 0.0


def calc_market_score() -> dict:
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
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    # SOX
    try:
        sox = yf.Ticker("^SOX").history(period="6d", auto_adjust=True)
        if sox is not None and not sox.empty and len(sox) >= 2:
            chg = float((sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1.0) * 100.0)
            score += float(np.clip(chg / 2.0, -5.0, 5.0))
            mkt["sox_5d"] = chg
    except Exception:
        pass

    # NVDA
    try:
        nv = yf.Ticker("NVDA").history(period="6d", auto_adjust=True)
        if nv is not None and not nv.empty and len(nv) >= 2:
            chg = float((nv["Close"].iloc[-1] / nv["Close"].iloc[0] - 1.0) * 100.0)
            score += float(np.clip(chg / 3.0, -4.0, 4.0))
            mkt["nvda_5d"] = chg
    except Exception:
        pass

    score = int(np.clip(round(score), 0, 100))
    mkt["score"] = score
    return mkt


def _volume_ratio(symbol: str, lookback_days: int = 40, ma_window: int = 20) -> float:
    """
    ⑩ TOPIX出来高比の proxy:
    ^TOPX に Volume が無いケースがあるため、TOPIX ETF 1306.T を利用。
    """
    try:
        df = yf.Ticker(symbol).history(period=f"{lookback_days}d", auto_adjust=False)
        if df is None or df.empty or "Volume" not in df.columns:
            return np.nan
        vol = df["Volume"].astype(float)
        if len(vol) < ma_window + 2:
            return np.nan
        v_now = float(vol.iloc[-1])
        v_ma = float(vol.rolling(ma_window).mean().iloc[-2])  # 前日までの平均で比較
        if not (np.isfinite(v_now) and np.isfinite(v_ma) and v_ma > 0):
            return np.nan
        return float(v_now / v_ma)
    except Exception:
        return np.nan


def _daily_chg(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="3d", auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return np.nan
        c = df["Close"].astype(float)
        return float((c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0)
    except Exception:
        return np.nan


def abnormal_day_flag() -> AbnormalFlag:
    """
    ⑩ 想定外日フラグ:
    - 1306.T 出来高が 20日平均の70%未満
    - NK=F が -1.5%以下（取得できる場合）
    """
    reasons = []
    flag = False

    vr = _volume_ratio("1306.T")
    if np.isfinite(vr) and vr < 0.70:
        flag = True
        reasons.append(f"TOPIX出来高proxy(1306.T) 比={vr:.2f} < 0.70")

    fut = _daily_chg("NK=F")
    if np.isfinite(fut) and fut <= -1.5:
        flag = True
        reasons.append(f"Nikkei先物proxy(NK=F) 前日比={fut:.2f}% <= -1.5%")

    return AbnormalFlag(flag=flag, reasons=reasons)
