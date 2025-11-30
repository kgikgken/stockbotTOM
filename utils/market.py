import yfinance as yf
import numpy as np

def safe_price(ticker: str):
    try:
        df = yf.Ticker(ticker).history(period="1d")
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except:
        return None


def calc_market_score() -> dict:
    score = 50  # 基準

    # 日経225
    nk = safe_price("^N225")
    if nk:
        if nk > 34000: score += 10
        if nk > 30000: score += 5
        if nk < 28000: score -= 5

    # NASDAQ
    nd_hist = yf.Ticker("^NDX").history(period="5d")
    if len(nd_hist) >= 2:
        pct = (nd_hist["Close"].iloc[-1] - nd_hist["Close"].iloc[-2]) / nd_hist["Close"].iloc[-2]
        if pct > 0.007: score += 6
        elif pct < -0.007: score -= 8

    # USDJPY
    usd = safe_price("JPY=X")
    if usd:
        if usd > 150: score -= 8
        elif usd > 147: score -= 4
        elif usd < 145: score += 4

    # VIX
    vix = safe_price("^VIX")
    if vix:
        if vix < 15: score += 6
        elif vix < 20: score += 2
        elif vix > 22: score -= 6
        elif vix > 30: score -= 12

    # 米10年
    us10 = safe_price("^TNX")
    if us10:
        if us10 > 4.6: score -= 6
        elif us10 < 4.0: score += 5

    score = int(np.clip(score, 0, 100))

    if score >= 70: comment = "強い地合い（攻め寄り）"
    elif score >= 55: comment = "やや強め（押し目◯）"
    elif score >= 45: comment = "中立"
    elif score >= 35: comment = "弱め（控えめ）"
    else: comment = "悪地合い（IN厳禁）"

    return {"score": score, "comment": comment}
