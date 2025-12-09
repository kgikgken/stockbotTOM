from __future__ import annotations

import numpy as np


# ==============================================
# 直近終値
# ==============================================
def _last_val(series):
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


# ==============================================
# 20日ボラ（ATRもどき）
# ==============================================
def calc_vola20(close):
    try:
        r = np.log(close / close.shift(1))
        return float(np.nanstd(r.tail(20)))
    except Exception:
        return np.nan


# ==============================================
# compute_rr (main.py が呼ぶ関数)
# ==============================================
def compute_rr(hist, mkt_score):
    """
    出力: dict
      rr:         R倍数
      entry:      エントリー価格
      tp_pct:     利確%（= profit target, +0.15 など）
      sl_pct:     損切り%（= stop loss, -0.05 など）
    """
    close = hist["Close"].astype(float)
    entry = _last_val(close)
    if not np.isfinite(entry):
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0)

    vola = calc_vola20(close)
    if not np.isfinite(vola) or vola <= 0:
        vola = 0.02  # デフォルトボラ

    # 地合いスコアを 0〜100 → -1〜+1 に正規化
    try:
        if isinstance(mkt_score, dict):
            s = float(mkt_score.get("score", 50))
        else:
            s = float(mkt_score)
    except Exception:
        s = 50.0

    strength = (s - 50.0) / 50.0
    strength = float(np.clip(strength, -1.0, 1.0))

    # ベース: stop = 1ATR, target = 3ATR
    base_stop = vola
    base_target = vola * 3.0

    # 地合いで TP/SL を調整（RR重視）
    stop_mul = 1.0 - 0.4 * strength   # 強いほどやや浅く
    target_mul = 1.0 + 0.8 * strength # 強いほどかなり伸ばす

    stop_mul = float(np.clip(stop_mul, 0.7, 1.3))
    target_mul = float(np.clip(target_mul, 0.7, 1.8))

    stop_pct = base_stop * stop_mul
    target_pct = base_target * target_mul

    if stop_pct <= 0:
        rr = 0.0
    else:
        rr = target_pct / stop_pct

    return dict(
        rr=float(rr),
        entry=float(entry),
        tp_pct=float(target_pct),
        sl_pct=float(-stop_pct),
    )