from __future__ import annotations

import numpy as np

from utils.features import atr_percent, setup_type


def universe_filter(ind: dict) -> tuple[bool, str]:
    """
    母集団：全銘柄OK、ただし “入らない理由” を明確化
    """
    c = float(ind.get("close", np.nan))
    adv20 = float(ind.get("adv20", np.nan))
    atrp = float(atr_percent(ind))

    # 価格レンジ（事故回避）
    if not (np.isfinite(c) and 200 <= c <= 15000):
        return False, "価格レンジ外"

    # 流動性
    if not (np.isfinite(adv20) and adv20 >= 200_000_000):
        return False, "流動性弱(ADV20<200M)"

    # ボラ
    if atrp < 1.5:
        return False, "ボラ不足(ATR%<1.5)"

    # 高ボラ事故ゾーン（除外に近い扱い）
    if atrp >= 6.0:
        return False, "事故ゾーン(ATR%>=6)"

    # Setup（A/Bのみ）
    st = setup_type(ind)
    if st not in ("A", "B"):
        return False, "形不一致"

    return True, ""


def gu_flag(hist) -> bool:
    """
    GU_flag = Open > PrevClose + 1.0ATR
    """
    try:
        if hist is None or len(hist) < 3:
            return False
        o = float(hist["Open"].iloc[-1])
        pc = float(hist["Close"].iloc[-2])
        # ATR proxy: TrueRange rollingはfeatures側で取ってるのでここは安全に近似
        atr = float((hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1])
        if not all(np.isfinite([o, pc, atr])) or atr <= 0:
            return False
        return bool(o > pc + 1.0 * atr)
    except Exception:
        return False