from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import score_stock, calc_inout_for_stock
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

EARNINGS_EXCLUDE_DAYS = 3

SWING_MAX_FINAL = 5
SWING_SCORE_MIN = 70.0
SWING_RR_MIN = 2.0
SWING_EV_R_MIN = 0.40

SECTOR_TOP_N = 5


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def fetch_history(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.4)
    return None


def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    out = []
    for _, r in df.iterrows():
        out.append({
            "label": str(r.get("label", "")).strip(),
            "date": str(r.get("date", "")).strip(),
            "time": str(r.get("time", "")).strip(),
            "datetime": str(r.get("datetime", "")).strip(),
        })
    return out


def build_event_warnings(today_date) -> Tuple[List[str], bool]:
    warns = []
    near = False
    for ev in load_events():
        dt = parse_event_datetime_jst(ev["datetime"], ev["date"], ev["time"])
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            near = True
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            warns.append(f"⚠ {ev['label']}（{dt.strftime('%Y-%m-%d %H:%M JST')} / {when}）")
    if not warns:
        warns.append("- 特になし")
    return warns, near


def expected_r_from_in_rank(in_rank: str, rr: float) -> float:
    win = {"強IN": 0.45, "通常IN": 0.40, "弱めIN": 0.33}.get(in_rank, 0.25)
    return win * rr - (1 - win)


def run_swing(today_date, mkt_score: int) -> List[Dict]:
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    t_col = "ticker" if "ticker" in uni.columns else "code"
    cands = []

    for _, r in uni.iterrows():
        ticker = str(r.get(t_col, "")).strip()
        if not ticker:
            continue

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 120:
            continue

        # --- 順張り必須条件（逆張り排除）
        close = hist["Close"]
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        if not (close.iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1]):
            continue

        score = score_stock(hist)
        if score < SWING_SCORE_MIN:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score)
        rr = rr_info["rr"]
        if rr < SWING_RR_MIN:
            continue

        ev = expected_r_from_in_rank(in_rank, rr)
        if ev < SWING_EV_R_MIN:
            continue

        price = _safe_float(close.iloc[-1])
        entry = rr_info["entry"]
        gap = (price / entry - 1) * 100 if entry > 0 else np.nan

        cands.append({
            "ticker": ticker,
            "name": r.get("name", ticker),
            "sector": r.get("sector", r.get("industry_big", "不明")),
            "score": score,
            "rr": rr,
            "ev": ev,
            "in_rank": in_rank,
            "entry": entry,
            "price": price,
            "gap": gap,
            "tp_pct": rr_info["tp_pct"],
            "sl_pct": rr_info["sl_pct"],
            "tp_price": rr_info["tp_price"],
            "sl_price": rr_info["sl_price"],
        })

    cands.sort(key=lambda x: (x["rr"], x["ev"], x["score"]), reverse=True)
    return cands[:SWING_MAX_FINAL]


def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return
    chunks = [text[i:i+3800] for i in range(0, len(text), 3800)]
    for c in chunks:
        r = requests.post(WORKER_URL, json={"text": c}, timeout=20)
        print("[LINE]", r.status_code)


def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    mkt_score = int(mkt.get("score", 50))

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt_score)

    events, event_near = build_event_warnings(today_date)
    sectors = top_sectors_5d(SECTOR_TOP_N)
    swing = run_swing(today_date, mkt_score)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append(f"◆ 地合い: {mkt_score}点 ({mkt.get('comment','')})\n")

    lines.append("📈 セクター（5日）")
    for i, (s, p) in enumerate(sectors):
        lines.append(f"{i+1}. {s} ({p:+.2f}%)")
    lines.append("")

    lines.append("⚠ イベント")
    lines.extend(events)
    lines.append("")

    lines.append("🏆 Swing")
    for c in swing:
        lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
        lines.append(f"  Score:{c['score']:.1f} RR:{c['rr']:.2f} EV:{c['ev']:.2f} IN:{c['in_rank']}")
        lines.append(f"  IN:{c['entry']:.1f} 現在:{c['price']:.1f} ({c['gap']:+.2f}%)")
        lines.append(f"  TP:{c['tp_pct']*100:+.1f}% ({c['tp_price']:.1f}) SL:{c['sl_pct']*100:+.1f}% ({c['sl_price']:.1f})")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text)

    report = "\n".join(lines)
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()