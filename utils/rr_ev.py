from __future__ import annotations


def calc_rr_ev_speed(ticker, setup, entry, market):
    atr = setup["atr"]
    entry_price = entry["entry"]

    # Stop / TP
    stop = entry_price - 1.2 * atr
    tp2 = entry_price + 3.0 * atr
    tp1 = entry_price + 1.5 * atr

    r = (tp2 - entry_price) / (entry_price - stop)

    if r < 1.8:
        return {"valid": False, "ticker": ticker, "reason": "RR不足"}

    # Pwin proxy（固定だが保守的）
    pwin = 0.45 if setup["setup_type"] == "A" else 0.40

    ev = pwin * r - (1 - pwin) * 1
    adj_ev = ev * market["regime_multiplier"]

    if adj_ev < 0.3:
        return {"valid": False, "ticker": ticker, "reason": "EV不足"}

    expected_days = (tp2 - entry_price) / atr
    r_per_day = r / expected_days

    if expected_days > 5 or r_per_day < 0.5:
        return {"valid": False, "ticker": ticker, "reason": "速度不足"}

    return {
        "valid": True,
        "rr": r,
        "ev": ev,
        "adj_ev": adj_ev,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "expected_days": expected_days,
        "r_per_day": r_per_day,
    }