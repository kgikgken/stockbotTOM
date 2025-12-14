from __future__ import annotations

import numpy as np
import pandas as pd


def _last_val(series: pd.Series) -> float:
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return np.nan
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    v = tr.rolling(period).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan


def compute_tp_sl_rr(hist: pd.DataFrame, mkt_score: int, for_day: bool = False) -> dict:
    """
    可変RR（固定RR禁止）
    - SL: 構造（直近安値）＋ATR
    - TP: 直近高値/抵抗（60d high）に現実的に届く範囲（※vABではTPは参考値）
    - entry: 「押し目基準」（MA20 - 0.5ATR）を軸に、現値を上回らないよう補正
    戻り:
      entry, tp_pct, sl_pct, tp_price, sl_price, rr, entry_basis, atr14
    """
    df = hist.copy()
    close = df["Close"].astype(float)
    price = _last_val(close)
    if not np.isfinite(price) or price <= 0:
        return dict(rr=0.0, entry=0.0, tp_pct=0.0, sl_pct=0.0, tp_price=0.0, sl_price=0.0, entry_basis="na", atr14=np.nan)

    atr14 = _atr(df, period=14)
    atr = atr14 if np.isfinite(atr14) and atr14 > 0 else max(price * 0.01, 1.0)

    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else price
    ma5 = float(close.rolling(5).mean().iloc[-1]) if len(close) >= 5 else price

    # entry（押し目基準）
    entry = ma20 - 0.5 * atr
    entry_basis = "pullback"

    # 強トレンドで浅め（entryは上げすぎない）
    if price > ma5 > ma20:
        entry = ma20 + (ma5 - ma20) * 0.25
        entry_basis = "trend_pullback"

    # 現値より上にはしない
    if entry > price:
        entry = price * 0.995

    # 構造SL（直近安値）
    lookback = 8 if for_day else 12
    swing_low = float(df["Low"].astype(float).tail(lookback).min())
    sl_price = min(entry - 0.8 * atr, swing_low - 0.2 * atr)

    # SLクランプ
    sl_pct = (sl_price / entry - 1.0)
    sl_pct = float(np.clip(sl_pct, -0.10, -0.02))
    sl_price = entry * (1.0 + sl_pct)

    # TP（抵抗帯：60日高値手前）
    hi_window = 60 if len(close) >= 60 else len(close)
    high_60 = float(close.tail(hi_window).max())
    tp_price = min(high_60 * 0.995, entry * (1.0 + (0.22 if not for_day else 0.08)))

    # 地合いでTP微調整（参考）
    if mkt_score >= 70:
        tp_price *= 1.03
    elif mkt_score <= 45:
        tp_price *= 0.97

    if tp_price <= entry:
        tp_price = entry * (1.0 + (0.06 if not for_day else 0.03))

    tp_pct = (tp_price / entry - 1.0)
    tp_pct = float(np.clip(tp_pct, 0.03, 0.30))
    tp_price = entry * (1.0 + tp_pct)

    rr = tp_pct / abs(sl_pct) if sl_pct < 0 else 0.0

    return dict(
        rr=float(rr),
        entry=float(round(entry, 1)),
        tp_pct=float(tp_pct),
        sl_pct=float(sl_pct),
        tp_price=float(round(tp_price, 1)),
        sl_price=float(round(sl_price, 1)),
        entry_basis=entry_basis,
        atr14=float(atr14) if np.isfinite(atr14) else np.nan,
    )
