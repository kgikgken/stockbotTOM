from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from utils.util import atr14, env_float, human_yen, round_to_tick, safe_float


SETUP_PRIORITY_TREND = ["A1-Strong", "A1", "B", "A2"]
SETUP_PRIORITY_LEADERS = ["A1-Strong", "A1", "B"]


def liquidity_filters(df: pd.DataFrame) -> Dict[str, float | str | bool]:
    if df is None or df.empty or len(df) < 30:
        return {
            "ok": False,
            "adv20": 0.0,
            "mdv20": 0.0,
            "dv_cv20": float("nan"),
            "grade": "thin",
            "score": 0.0,
            "label": "-",
        }

    traded_value = df["Close"].astype(float) * df["Volume"].astype(float)
    adv20 = float(traded_value.tail(20).mean())
    mdv20 = float(traded_value.tail(20).median())
    dv_cv20 = float(traded_value.tail(20).std(ddof=0) / traded_value.tail(20).mean()) if traded_value.tail(20).mean() > 0 else float("nan")
    strict_min = env_float("LIQ_STRICT_ADV_MIN", 800e6)
    relax_min = env_float("LIQ_RELAX_ADV_MIN", 500e6)
    if adv20 >= strict_min and mdv20 >= strict_min * 0.70:
        grade = "thick"
        score = 1.0
    elif adv20 >= relax_min and mdv20 >= relax_min * 0.65:
        grade = "ok"
        score = 0.72
    else:
        grade = "thin"
        score = 0.35 if adv20 >= relax_min * 0.7 else 0.0
    return {
        "ok": grade != "thin",
        "adv20": adv20,
        "mdv20": mdv20,
        "dv_cv20": dv_cv20,
        "grade": grade,
        "score": score,
        "label": f"ADV {human_yen(adv20)}",
    }


def _plan(
    setup: str,
    lane: str,
    close: float,
    atr: float,
    pivot: float,
    base_ma: float,
    stop_anchor: float,
    pullback_low: float,
) -> Dict | None:
    if not (np.isfinite(close) and np.isfinite(atr) and np.isfinite(base_ma) and close > 0 and atr > 0):
        return None
    tick = close
    if setup == "B":
        entry = round_to_tick(max(pivot, close), tick, mode="up")
        stop = round_to_tick(min(stop_anchor, pullback_low) - 0.35 * atr, tick, mode="down")
        entry_type = "stop"
        entry_lo = entry
        entry_hi = entry
        status = "逆指待ち"
    else:
        raw_entry = max(base_ma, close - 0.45 * atr)
        entry = round_to_tick(min(close, raw_entry), tick)
        entry_lo = round_to_tick(entry - 2 * 1.0, tick, mode="down")
        entry_hi = round_to_tick(entry + 2 * 1.0, tick, mode="up")
        stop = round_to_tick(min(stop_anchor, pullback_low) - 0.30 * atr, tick, mode="down")
        entry_type = "limit"
        status = "指値待ち"

    if not (np.isfinite(entry) and np.isfinite(stop) and entry > stop > 0):
        return None
    risk_pct = (entry - stop) / entry * 100.0
    if risk_pct <= 0 or risk_pct > env_float("MAX_RISK_PCT", 8.0):
        return None
    tp1 = entry + 2.0 * (entry - stop)
    tp2 = entry + 3.0 * (entry - stop)
    rr = (tp2 - entry) / (entry - stop)
    order_text = f"逆指 {entry:,.0f}" if entry_type == "stop" else f"指値 {entry_lo:,.0f}-{entry_hi:,.0f}"
    return {
        "setup": setup,
        "lane": lane,
        "entry_type": entry_type,
        "entry": float(entry),
        "entry_lo": float(entry_lo),
        "entry_hi": float(entry_hi),
        "stop": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "rr": float(rr),
        "risk_pct": float(risk_pct),
        "status": status,
        "order_text": order_text,
    }


def build_setup_info(df: pd.DataFrame, lane: str = "trend") -> List[Dict]:
    if df is None or df.empty or len(df) < 220:
        return []

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    atr_s = atr14(df)
    atr = safe_float(atr_s.iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return []

    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    last = safe_float(close.iloc[-1])
    m10 = safe_float(ma10.iloc[-1])
    m20 = safe_float(ma20.iloc[-1])
    m50 = safe_float(ma50.iloc[-1])
    m150 = safe_float(ma150.iloc[-1])
    m200 = safe_float(ma200.iloc[-1])
    recent_high10 = safe_float(high.iloc[-11:-1].max())
    recent_high20 = safe_float(high.iloc[-21:-1].max())
    recent_low10 = safe_float(low.tail(10).min())
    recent_low20 = safe_float(low.tail(20).min())
    recent_low5 = safe_float(low.tail(5).min())

    if not all(np.isfinite(x) for x in [last, m10, m20, m50, m150, m200, recent_high10, recent_high20, recent_low10, recent_low20, recent_low5]):
        return []

    pullback_20 = (recent_high20 - last) / recent_high20 * 100.0 if recent_high20 > 0 else float("nan")
    tight_range10 = (recent_high10 - recent_low10) / last * 100.0 if last > 0 else float("nan")
    tight_range20 = (recent_high20 - recent_low20) / last * 100.0 if last > 0 else float("nan")
    ma200_up = m200 >= safe_float(ma200.shift(20).iloc[-1], m200)
    plans: List[Dict] = []

    # A1-Strong: very strong stacked trend with shallow pullback.
    cond_a1s = (
        last > m10 > m20 > m50 > m150 > m200
        and ma200_up
        and pullback_20 <= env_float("A1S_MAX_PULLBACK_PCT", 8.0)
        and recent_low5 <= m20 * 1.015
    )
    if cond_a1s:
        plan = _plan(
            "A1-Strong",
            lane,
            close=last,
            atr=atr,
            pivot=recent_high10,
            base_ma=m20,
            stop_anchor=min(m50, recent_low10),
            pullback_low=recent_low10,
        )
        if plan:
            plans.append(plan)

    # A1: normal strong-trend pullback.
    cond_a1 = (
        last > m20 > m50 > m150 > m200
        and pullback_20 <= env_float("A1_MAX_PULLBACK_PCT", 12.0)
        and recent_low10 <= m20 * 1.04
    )
    if cond_a1:
        plan = _plan(
            "A1",
            lane,
            close=last,
            atr=atr,
            pivot=recent_high10,
            base_ma=m20,
            stop_anchor=min(m50, recent_low20),
            pullback_low=recent_low10,
        )
        if plan:
            plans.append(plan)

    # A2: earlier-stage pullback, disabled for leaders by default.
    allow_a2 = lane == "trend" or str(env_float("LEADERS_ALLOW_A2", 0.0)) in {"1.0", "1"}
    cond_a2 = (
        allow_a2
        and last > m50 > m150 > m200
        and pullback_20 <= env_float("A2_MAX_PULLBACK_PCT", 15.0)
        and recent_low10 <= m50 * 1.03
    )
    if cond_a2:
        plan = _plan(
            "A2",
            lane,
            close=last,
            atr=atr,
            pivot=recent_high20,
            base_ma=m50,
            stop_anchor=min(recent_low20, m150),
            pullback_low=recent_low20,
        )
        if plan:
            plans.append(plan)

    # B: contraction / breakout near pivot.
    cond_b = (
        last > m20 > m50
        and recent_high20 >= recent_high10
        and tight_range10 <= env_float("B_MAX_RANGE10_PCT", 6.0)
        and tight_range20 <= env_float("B_MAX_RANGE20_PCT", 12.0)
        and last >= recent_high20 * env_float("B_NEAR_PIVOT_MIN", 0.96)
    )
    if cond_b:
        plan = _plan(
            "B",
            lane,
            close=last,
            atr=atr,
            pivot=recent_high20 * 1.001,
            base_ma=m20,
            stop_anchor=min(m20, recent_low10),
            pullback_low=recent_low10,
        )
        if plan:
            plans.append(plan)

    priority = SETUP_PRIORITY_LEADERS if lane == "leaders" else SETUP_PRIORITY_TREND
    order = {name: i for i, name in enumerate(priority)}
    plans.sort(key=lambda x: order.get(str(x.get("setup")), 999))
    return plans
