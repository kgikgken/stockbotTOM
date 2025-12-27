import pandas as pd
import yfinance as yf
import numpy as np

from utils.features import add_features
from utils.setup import detect_setup
from utils.entry import decide_action
from utils.rr_ev import calc_rr_ev
from utils.diversify import apply_sector_limit


def run_swing_screener(universe_path, today_date, market_score, delta_3d, top_sectors):
    dfu = pd.read_csv(universe_path)
    tcol = "ticker" if "ticker" in dfu.columns else "code"

    cands = []

    for _, r in dfu.iterrows():
        t = str(r[tcol])
        sec = r.get("sector", r.get("industry_big", "不明"))

        if sec not in top_sectors:
            continue

        try:
            h = yf.Ticker(t).history(period="260d", auto_adjust=True)
        except Exception:
            continue
        if len(h) < 120:
            continue

        h = add_features(h)
        adv = h["adv20"].iloc[-1]
        if adv < 200_000_000:
            continue

        setup = detect_setup(h)
        if setup != "A":
            continue

        price = h["Close"].iloc[-1]
        atr = h["atr"].iloc[-1]
        entry = h["ma20"].iloc[-1]

        stop = entry - 1.2 * atr
        tp2 = entry + 3.0 * (entry - stop)

        r, ev = calc_rr_ev(entry, stop, tp2, pwin=0.45)

        exp_days = (tp2 - entry) / (0.9 * atr)
        r_day = r / exp_days if exp_days > 0 else 0

        action = decide_action(price, entry, atr, gu=False)

        cands.append({
            "ticker": t,
            "sector": sec,
            "setup": setup,
            "entry": entry,
            "price": price,
            "stop": stop,
            "tp2": tp2,
            "rr": r,
            "ev": ev,
            "r_day": r_day,
            "action": action,
        })

    cands.sort(key=lambda x: x["r_day"], reverse=True)
    cands = apply_sector_limit(cands)
    return cands[:5]