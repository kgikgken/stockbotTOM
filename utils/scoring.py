from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from rr import compute_rr_block

# =========================
def _atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().iloc[-1]

def _ma(s, n):
    return s.rolling(n).mean()

# =========================
def screen_swing(today_date, mkt_score: int):
    uni = pd.read_csv("universe_jpx.csv")
    out = []

    for _, r in uni.iterrows():
        t = str(r["ticker"])
        try:
            df = yf.Ticker(t).history(period="200d", auto_adjust=True)
            if len(df) < 120:
                continue
        except Exception:
            continue

        c = df["Close"]
        atr = _atr(df)
        ma20, ma50 = _ma(c, 20), _ma(c, 50)

        # --- トレンド押し目 only ---
        if not (c.iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1]):
            continue
        if ma20.iloc[-1] - ma20.iloc[-6] <= 0:
            continue

        entry = ma20.iloc[-1]
        in_low = entry - 0.5 * atr
        in_high = entry + 0.5 * atr

        gu = df["Open"].iloc[-1] > df["Close"].iloc[-2] + atr

        rr_block = compute_rr_block(df, entry)
        if rr_block is None:
            continue

        rr, stop, tp1, tp2 = rr_block
        if rr < 2.2:
            continue

        exp_days = (tp2 - entry) / max(atr, 1)
        r_per_day = rr / max(exp_days, 1)

        ev = 0.42 * rr - 0.58
        if ev < 0.4:
            continue
        if exp_days > 5:
            continue

        price_now = c.iloc[-1]
        action = (
            "即IN可" if (not gu and abs(price_now-entry)/atr <= 0.3)
            else "指値待ち"
        )

        out.append(dict(
            ticker=t,
            name=r.get("name", t),
            sector=r.get("sector", "不明"),
            setup="A",
            rr=rr,
            ev=ev,
            r_per_day=r_per_day,
            entry=entry,
            in_low=in_low,
            in_high=in_high,
            price_now=price_now,
            atr=atr,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            exp_days=exp_days,
            gu=gu,
            action=action
        ))

    out.sort(key=lambda x: (x["ev"], x["r_per_day"]), reverse=True)
    return out[:5]