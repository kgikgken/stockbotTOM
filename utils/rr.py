from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from utils.scoring import calc_in_zone


def _last(s: pd.Series) -> float:
    try:
        return float(s.iloc[-1])
    except Exception:
        return float("nan")


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return float("nan")
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    pc = c.shift(1)

    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else float("nan")


def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int, setup: str) -> Dict[str, float]:
    if hist is None or len(hist) < 80:
        return dict(rr=0.0)

    df = hist.copy()
    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float) if "Open" in df.columns else close
    low = df["Low"].astype(float)

    price = _last(close)
    prev_close = float(close.shift(1).iloc[-1]) if len(close) >= 2 else float("nan")
    open_last = _last(open_)

    atr = _atr(df, 14)
    if not np.isfinite(atr) or atr <= 0:
        atr = max(float(price) * 0.01 if np.isfinite(price) else 1.0, 1.0)

    inz = calc_in_zone(df, setup)
    in_center = float(inz["center"])
    in_lower = float(inz["lower"])
    in_upper = float(inz["upper"])

    entry = in_center
    if np.isfinite(price) and entry > price:
        entry = price * 0.995

    gu_flag = False
    if np.isfinite(open_last) and np.isfinite(prev_close):
        if open_last > prev_close + 1.0 * atr:
            gu_flag = True

    if setup == "A":
        stop = in_lower - 0.7 * atr
        swing_low = float(low.tail(12).min())
        stop = min(stop, swing_low - 0.2 * atr)
    else:
        stop = in_center - 1.0 * atr
        swing_low = float(low.tail(10).min())
        stop = min(stop, swing_low - 0.2 * atr)

    risk = entry - stop
    if not np.isfinite(risk) or risk <= 0:
        return dict(rr=0.0)

    min_risk = max(entry * 0.02, 0.7 * atr)
    if risk < min_risk:
        stop = entry - min_risk
        risk = entry - stop

    # --- Targets ---
    # TP1/TP2 は「Rベース」だが、上値余地（抵抗）で現実的にクランプする。
    tp1_raw = entry + 1.5 * risk
    tp2_raw = entry + 3.0 * risk

    hi_window = 60 if len(close) >= 60 else len(close)
    high_60 = float(close.tail(hi_window).max())
    resistance = high_60 * 0.995  # 抵抗手前

    # 強い上値余地がある時は 3R を超えるのも許容（ただし抵抗まで）
    rr_cap_max = 6.0
    tp2_cap_by_rr = entry + rr_cap_max * risk

    # 抵抗が遠いほど tp2 は伸びる（最大 rr_cap_max まで）。抵抗が近ければ自然に RR が落ちる。
    tp2 = min(resistance, tp2_cap_by_rr)
    tp1 = min(tp1_raw, tp2)

    R = (tp2 - entry) / risk if risk > 0 else 0.0
    # 追いかけは「上方向」だけ（上に離れているほど危険）。下方向は“落ちるナイフ”として別扱い。
    dist_above_atr = ((price - in_center) / atr) if (np.isfinite(price) and atr > 0 and price > in_center) else 0.0
    dist_below_atr = ((in_center - price) / atr) if (np.isfinite(price) and atr > 0 and price < in_center) else 0.0

    return dict(
        entry=float(round(entry, 1)),
        in_lower=float(round(in_lower, 1)),
        in_upper=float(round(in_upper, 1)),
        stop=float(round(stop, 1)),
        tp1=float(round(tp1, 1)),
        tp2=float(round(tp2, 1)),
        rr=float(R),
        atr=float(round(atr, 1)),
        gu_flag=bool(gu_flag),
        dist_above_atr=float(dist_above_atr),
        dist_below_atr=float(dist_below_atr),
        in_dist_atr=float(dist_above_atr),
    )
