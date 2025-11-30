import yfinance as yf
import numpy as np

# ============================================================
# 地合いスコアを計算する
# ============================================================

def safe_price(ticker: str):
    """値取得失敗時は None を返す"""
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if len(data) == 0:
            return None
        return float(data["Close"].iloc[-1])
    except:
        return None


def calc_market_score() -> dict:
    """
    地合いスコア（0〜100）
    内訳スコア付き
    """

    score = 50  # 基準（中立）

    # ============================
    # ① 日経先物（日中） 
    # ============================
    nk = safe_price("^N225")
    if nk:
        if nk > 30000:
            score += 5
        if nk > 34000:
            score += 8
        if nk < 28000:
            score -= 5

    # ============================
    # ② NASDAQ先物
    # ============================
    nd = safe_price("^NDX")
    if nd:
        # 直近2日の差分
        nd_hist = yf.Ticker("^NDX").history(period="5d")
        if len(nd_hist) >= 2:
            pct = (nd_hist["Close"].iloc[-1] - nd_hist["Close"].iloc[-2]) / nd_hist["Close"].iloc[-2]
            if pct > 0.005:
                score += 6
            elif pct < -0.005:
                score -= 8

    # ============================
    # ③ USDJPY（リスクオン/オフ）
    # ============================
    usd = safe_price("JPY=X")
    if usd:
        if usd > 150:
            score -= 8   # 円安すぎ → 日本株マイナス
        elif usd > 147:
            score -= 4
        elif usd < 145:
            score += 4   # 円高 → グロース有利

    # ============================
    # ④ VIX（恐怖指数）
    # ============================
    vix = safe_price("^VIX")
    if vix:
        if vix < 15:
            score += 6
        elif vix < 20:
            score += 2
        elif vix > 22:
            score -= 6
        elif vix > 30:
            score -= 10

    # ============================
    # ⑤ 米10年金利
    # ============================
    us10y = safe_price("^TNX")  # %
    if us10y:
        if us10y > 4.5:
            score -= 5
        elif us10y < 4.0:
            score += 5

    # スコア範囲固定
    score = int(np.clip(score, 0, 100))

    # コメント作成
    if score >= 70:
        comment = "強い地合い（攻め寄り）"
    elif score >= 55:
        comment = "やや強め（押し目狙い◯）"
    elif score >= 45:
        comment = "中立（慎重に）"
    elif score >= 35:
        comment = "やや弱め（ロット控えめ）"
    else:
        comment = "弱い地合い（無理IN禁止）"

    return {
        "score": score,
        "comment": comment
    }