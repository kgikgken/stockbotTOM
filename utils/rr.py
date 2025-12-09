import numpy as np

def _last_val(series):
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def calc_vola20(close):
    try:
        r = np.log(close / close.shift(1))
        return float(np.nanstd(r.tail(20)))
    except Exception:
        return np.nan


def _mkt_adjust(tp, sl, mkt_score: float):
    ms = float(mkt_score)
    # 中立=0とみなす
    # 想定: 0~100
    # 50を中立点
    bias = (ms - 50.0) / 50.0  # -1 ~ +1
    adj = 1.0 + 0.3 * bias     # 0.7 ~ 1.3
    return tp * adj, sl * adj


def compute_tp_sl_rr(hist, mkt_score: float):
    close = hist["Close"]
    entry = _last_val(close)
    if not np.isfinite(entry):
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0)

    vola = calc_vola20(close)
    if not np.isfinite(vola) or vola <= 0:
        vola = 0.02

    base_sl = 1.0 * vola
    base_tp = 3.0 * vola

    tp, sl = _mkt_adjust(base_tp, base_sl, mkt_score)
    rr = tp / sl if sl > 0 else 0.0

    return dict(
        rr=float(rr),
        entry=float(entry),
        tp_pct=float(tp),
        sl_pct=float(-sl),
    )