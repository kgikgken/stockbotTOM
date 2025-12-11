　import numpy as np
import yfinance as yf


def _chg_5d(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period="6d")
        if df is None or df.empty or len(df) < 2:
            return 0.0
        return float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100
    except Exception:
        return 0.0


def enhance_market_score() -> dict:
    nk = _chg_5d("^N225")
    tp = _chg_5d("^TOPX")

    base = (nk + tp) / 2.0
    score = 50 + np.clip(base, -20, 20)

    # --- SOX補正 ---
    sox = _chg_5d("^SOX")
    score += np.clip(sox / 2.0, -5, 5)

    # --- NVDA補正 ---
    nv = _chg_5d("NVDA")
    score += np.clip(nv / 3.0, -4, 4)

    score = int(np.clip(round(score), 0, 100))

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

    return {"score": score, "comment": comment}