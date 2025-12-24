from __future__ import annotations

import numpy as np
import yfinance as yf


def _five_day_chg(symbol: str) -> float:
    try:
        df = yf.Ticker(symbol).history(period='6d', auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return 0.0
        c = df['Close'].astype(float)
        return float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0)
    except Exception:
        return 0.0


def calc_market_score() -> dict:
    nk = _five_day_chg('^N225')
    tp = _five_day_chg('^TOPX')

    base = 50.0 + float(np.clip((nk + tp) / 2.0, -20, 20))
    score = int(np.clip(round(base), 0, 100))

    if score >= 70:
        comment = '強め'
    elif score >= 60:
        comment = 'やや強め'
    elif score >= 50:
        comment = '中立'
    elif score >= 40:
        comment = '弱め'
    else:
        comment = '弱い'

    return {'score': score, 'comment': comment, 'n225_5d': nk, 'topix_5d': tp}


def enhance_market_score() -> dict:
    mkt = calc_market_score()
    score = float(mkt.get('score', 50))

    try:
        sox = _five_day_chg('^SOX')
        score += float(np.clip(sox / 2.0, -5.0, 5.0))
        mkt['sox_5d'] = sox
    except Exception:
        pass

    try:
        nv = _five_day_chg('NVDA')
        score += float(np.clip(nv / 3.0, -4.0, 4.0))
        mkt['nvda_5d'] = nv
    except Exception:
        pass

    mkt['score'] = int(np.clip(round(score), 0, 100))
    return mkt


def market_momentum_3d() -> int:
    try:
        df = yf.Ticker('^TOPX').history(period='10d', auto_adjust=True)
        if df is None or df.empty or len(df) < 5:
            return 0
        close = df['Close'].astype(float)
        today = float(close.iloc[-1])
        prev = float(close.iloc[-4])  # 3営業日前近似
        chg = (today / prev - 1.0) * 100.0
        return int(np.clip(round(chg), -20, 20))
    except Exception:
        return 0
