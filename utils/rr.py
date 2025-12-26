def compute_tp_sl_rr(hist, mkt_score: int, for_day: bool = False) -> dict:
    df = hist.copy()
    close = df["Close"].astype(float)
    price = float(close.iloc[-1])

    atr = _atr(df, 14)
    if not atr or atr <= 0:
        atr = price * 0.01

    ma20 = close.rolling(20).mean().iloc[-1]
    ma5 = close.rolling(5).mean().iloc[-1]

    entry = ma20 - 0.5 * atr
    entry_basis = "pullback"

    if price > ma5 > ma20:
        entry = ma20 + (ma5 - ma20) * 0.25
        entry_basis = "trend_pullback"

    if entry > price:
        entry = price * 0.995

    lookback = 8 if for_day else 12
    swing_low = df["Low"].astype(float).tail(lookback).min()
    sl_price = min(entry - 0.8 * atr, swing_low - 0.2 * atr)

    sl_pct = (sl_price / entry - 1.0)
    sl_pct = float(np.clip(sl_pct, -0.10, -0.02))
    sl_price = entry * (1.0 + sl_pct)

    hi_window = 60 if len(close) >= 60 else len(close)
    high_60 = float(close.tail(hi_window).max())
    tp_price = min(high_60 * 0.995, entry * (1.0 + (0.22 if not for_day else 0.08)))

    if mkt_score >= 70:
        tp_price *= 1.03
    elif mkt_score <= 45:
        tp_price *= 0.97

    if tp_price <= entry:
        tp_price = entry * (1.0 + (0.06 if not for_day else 0.03))

    tp_pct = float(tp_price / entry - 1.0)
    tp_pct = float(np.clip(tp_pct, 0.03, 0.30))
    tp_price = entry * (1.0 + tp_pct)

    rr = tp_pct / abs(sl_pct) if sl_pct < 0 else 0.0

    return {
        "rr": float(rr),
        "entry": float(round(entry, 1)),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "tp_price": float(round(tp_price, 1)),
        "sl_price": float(round(sl_price, 1)),
        "entry_basis": entry_basis,
    }