from __future__ import annotations
import numpy as np


def _last(series):
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _vola20(close):
    try:
        r = np.log(close / close.shift(1))
        v = float(np.nanstd(r.tail(20)))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _base_tp_sl(vola):
    if not np.isfinite(vola):
        return 0.06, -0.03

    if vola < 0.015:
        return 0.06, -0.03
    elif vola < 0.03:
        return 0.08, -0.04
    else:
        return 0.12, -0.06


def _mkt_adjust(tp, sl, mkt_score):
    ms = int(mkt_score)

    if ms >= 70:
        tp *= 1.08
    elif ms >= 60:
        tp *= 1.04
    elif ms >= 50:
        tp *= 1.00
    elif ms >= 40:
        tp *= 0.93
        sl = max(sl, -0.03)
    else:
        tp *= 0.88
        sl = max(sl, -0.03)

    return float(tp), float(sl)


def _pullback_coef(hist):
    close = hist["Close"].astype(float)
    rsi = close.diff().clip(lower=0).rolling(14).mean() / \
          (-close.diff().clip(upper=0).rolling(14).mean() + 1e-9)
    rsi = 100 - 100 / (1 + rsi)
    r_last = float(rsi.iloc[-1]) if np.isfinite(rsi.iloc[-1]) else 50.0

    if len(close) >= 60:
        high = float(close.tail(60).max())
        off = (float(close.iloc[-1]) - high) / high
    else:
        off = 0.0

    if 30 <= r_last <= 45 and -0.18 <= off <= -0.05:
        coef = 1.3
    elif 40 <= r_last <= 60 and -0.15 <= off <= 0.03:
        coef = 1.1
    elif r_last < 30 or r_last > 70:
        coef = 0.7
    else:
        coef = 1.0

    return float(coef)


def _entry_price(hist):
    return _last(hist["Close"])


# ==============================================
# export
# ==============================================
def compute_tp_sl_rr(hist, mkt_score):
    close = hist["Close"]
    entry = _entry_price(hist)

    vola = _vola20(close)
    tp, sl = _base_tp_sl(vola)
    tp, sl = _mkt_adjust(tp, sl, mkt_score)

    coef = _pullback_coef(hist)
    tp *= coef

    rr = tp / abs(sl) if abs(sl) > 1e-9 else 0.0

    return {
        "entry": float(entry),
        "tp_pct": float(tp),
        "sl_pct": float(sl),
        "rr": float(rr),
    }