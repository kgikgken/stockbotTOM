import numpy as np


def _last_val(series):
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def calc_vola20(close):
    """20日ボラ（対数リターンの標準偏差）"""
    try:
        r = np.log(close / close.shift(1))
        return float(np.nanstd(r.tail(20)))
    except Exception:
        return np.nan


def compute_rr(hist, mkt_score):
    """
    出力: dict
      rr:     R倍数
      entry:  エントリー価格
      tp_pct: 利確% （+0.12 なら +12%）
      sl_pct: 損切り% （-0.04 なら -4%）
    """
    close = hist["Close"]
    entry = _last_val(close)
    if not np.isfinite(entry):
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0)

    vola = calc_vola20(close)
    if not np.isfinite(vola) or vola <= 0:
        vola = 0.02

    # シンプル版: stop = 1ATR, target = 3ATR → RR=3
    stop_pct = 1.0 * vola
    target_pct = 3.0 * vola

    rr = target_pct / stop_pct if stop_pct > 0 else 0.0

    return dict(
        rr=float(rr),
        entry=float(entry),
        tp_pct=float(target_pct),
        sl_pct=float(-stop_pct),
    )