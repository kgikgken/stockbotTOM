import numpy as np
import pandas as pd

# ===============================
# 安全取得
# ===============================
def _last(series):
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan

# ===============================
# ATR風のボラ20
# ===============================
def _vola20(close: pd.Series) -> float:
    try:
        r = np.log(close / close.shift(1))
        return float(np.nanstd(r.tail(20)))
    except Exception:
        return np.nan

# ===============================
# entry: 未来の押し目でIN
# ===============================
def _entry_price(close: pd.Series, ma20: float, atr: float) -> float:
    """
    entry = MA20 - 0.5ATR
    MA20が無ければ現値
    """
    cur = _last(close)

    if np.isfinite(ma20) and np.isfinite(atr) and atr > 0:
        ent = float(ma20 - atr * 0.5)
        # entryが現値より上に行くことは避ける
        if ent > cur:
            ent = cur * 0.995
        return ent
    return cur

# ===============================
# compute_rr
# ===============================
def compute_rr(hist: pd.DataFrame, mkt_score: float) -> dict:
    """
    returns:
      rr        = TP/SL ratio
      entry     = ideal entry price
      tp_pct    = +X%
      sl_pct    = -Y%
    """
    close = hist["Close"].astype(float)
    ma20 = hist["Close"].rolling(20).mean().iloc[-1]
    atr = _vola20(close)
    cur = _last(close)

    if not np.isfinite(cur):
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0)

    if not np.isfinite(atr) or atr <= 0:
        atr = 0.02  # fallback

    # entry = 未来の押し目
    entry = _entry_price(close, ma20, atr)

    # TP/SL (ATR×倍率)
    # 地合いによって伸縮
    # mkt_score 50基準
    stretch = (mkt_score - 50) / 100.0  # -0.5〜+0.5
    base_tp = 3.0 + stretch
    base_sl = 1.0 - stretch*0.5  # 地合い弱→浅く、強→深く

    if base_sl < 0.8:
        base_sl = 0.8
    if base_tp < 2.0:
        base_tp = 2.0

    tp_pct = base_tp * atr         # +X
    sl_pct = base_sl * atr * (-1)  # -Y

    # RR
    rr = abs(tp_pct / sl_pct) if sl_pct != 0 else 0.0

    return dict(
        rr=float(rr),
        entry=float(entry),
        tp_pct=float(tp_pct),
        sl_pct=float(sl_pct),
    )