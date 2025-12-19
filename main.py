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

# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

EARNINGS_EXCLUDE_DAYS = 3

SWING_MAX_FINAL = 3
SWING_SCORE_MIN = 70.0
SWING_RR_MIN = 1.8
SWING_EV_R_MIN = 0.40

SECTOR_TOP_N = 5

# ============================================================
def fetch_history(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None


def expected_r_from_in_rank(in_rank: str, rr: float) -> float:
    if rr <= 0:
        return -999.0
    win = {"強IN": 0.45, "通常IN": 0.40, "弱めIN": 0.33}.get(in_rank, 0.25)
    return win * rr - (1.0 - win)


# ============================================================
def run_swing(today_date, mkt_score: int) -> List[Dict]:
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    t_col = "ticker" if "ticker" in uni.columns else "code"
    cands = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 120:
            continue

        score = score_stock(hist)
        if score is None or score < SWING_SCORE_MIN:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score)
        rr = rr_info["rr"]
        if rr < SWING_RR_MIN:
            continue

        ev = expected_r_from_in_rank(in_rank, rr)
        if ev < SWING_EV_R_MIN:
            continue

        cands.append({
            "ticker": ticker,
            "name": row.get("name", ticker),
            "sector": row.get("sector", "不明"),
            "score": score,
            "rr": rr,
            "ev": ev,
            "in_rank": in_rank,
            **rr_info
        })

    cands.sort(key=lambda x: (x["score"], x["ev"], x["rr"]), reverse=True)
    return cands[:SWING_MAX_FINAL]


# ============================================================
def build_report(today_str, today_date, mkt, pos_text, total_asset):
    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append(f"- 地合い: {mkt['score']}点 ({mkt['comment']})\n")

    sectors = top_sectors_5d(SECTOR_TOP_N)
    lines.append("📈 セクター（5日）")
    for i, (s, p) in enumerate(sectors, 1):
        lines.append(f"{i}. {s} ({p:+.2f}%)")
    lines.append("")

    swing = run_swing(today_date, mkt["score"])
    lines.append("🏆 Swing（数日〜2週）Core候補")
    if not swing:
        lines.append("- 該当なし")
    for c in swing:
        lines.append(
            f"- {c['ticker']} {c['name']} [{c['sector']}]\n"
            f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R IN:{c['in_rank']} EV:{c['ev']:.2f}R\n"
            f"  IN:{c['entry']:.1f} TP:{c['tp_price']:.1f} SL:{c['sl_price']:.1f}"
        )

    lines.append("\n📊 ポジション")
    lines.append(pos_text)

    return "\n".join(lines)


# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return
    requests.post(WORKER_URL, json={"text": text}, timeout=20)


# ============================================================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score=mkt["score"])

    report = build_report(today_str, today_date, mkt, pos_text, total_asset)
    print(report)
    send_line(report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", repr(e))