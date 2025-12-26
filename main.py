from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score, calc_market_delta3d
from utils.sector import top_sectors_5d
from utils.scoring import (
    universe_filter,
    detect_setup,
    calc_in_zone,
    calc_action,
    estimate_pwin,
)
from utils.rr import compute_trade_plan
from utils.position import load_positions, analyze_positions

# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

MAX_FINAL = 5
SECTOR_TOP_N = 5

# ============================================================
# データ取得
# ============================================================
def fetch_history(ticker: str, period="300d") -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if df is not None and len(df) >= 120:
            return df
    except Exception:
        pass
    return None

# ============================================================
# EV計算
# ============================================================
def calc_ev(pwin: float, r: float) -> float:
    return pwin * r - (1.0 - pwin)

# ============================================================
# Swing Screener（World v2.0）
# ============================================================
def run_swing(today_date, mkt: dict) -> List[Dict]:
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    ticker_col = "ticker" if "ticker" in uni.columns else "code"

    # --- 地合い判定 ---
    mkt_score = mkt["score"]
    delta3d = calc_market_delta3d()

    no_trade = (
        mkt_score < 45 or
        (delta3d <= -5 and mkt_score < 55)
    )

    if no_trade:
        return [{"NO_TRADE": True, "reason": f"MarketScore={mkt_score}, Δ3d={delta3d}"}]

    # --- セクター ---
    top_sectors = [s for s, _ in top_sectors_5d(SECTOR_TOP_N)]

    cands = []

    for _, row in uni.iterrows():
        ticker = str(row.get(ticker_col, "")).strip()
        if not ticker:
            continue

        sector = str(row.get("sector", row.get("industry_big", "不明")))
        if sector not in top_sectors:
            continue

        hist = fetch_history(ticker)
        if hist is None:
            continue

        # --- Universe Filter ---
        if not universe_filter(hist):
            continue

        # --- Setup 判定 ---
        setup = detect_setup(hist)
        if setup not in ("A", "B"):
            continue

        # --- INゾーン ---
        in_zone = calc_in_zone(hist, setup)

        # --- Trade Plan ---
        plan = compute_trade_plan(hist, setup, mkt_score)
        if plan["R"] < 2.2:
            continue

        # --- Pwin / EV ---
        pwin = estimate_pwin(hist, sector_rank=top_sectors.index(sector) + 1)
        ev = calc_ev(pwin, plan["R"])
        if ev < 0.4:
            continue

        action = calc_action(hist, in_zone)

        cands.append(
            dict(
                ticker=ticker,
                name=row.get("name", ticker),
                sector=sector,
                setup=setup,
                **plan,
                pwin=pwin,
                ev=ev,
                action=action,
            )
        )

    cands.sort(key=lambda x: x["ev"], reverse=True)
    return cands[:MAX_FINAL]

# ============================================================
# レポート
# ============================================================
def build_report(today_str, today_date, mkt, pos_text):
    swing = run_swing(today_date, mkt)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")

    lines.append("◆ 今日の結論（Swing専用）")
    lines.append(f"- 地合い: {mkt['score']}点")
    lines.append(f"- ΔMarketScore_3d: {calc_market_delta3d()}")
    lines.append("")

    lines.append("📈 セクター（5日）")
    for i, (s, p) in enumerate(top_sectors_5d(SECTOR_TOP_N), 1):
        lines.append(f"{i}. {s} ({p:+.2f}%)")
    lines.append("")

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")

    if swing and "NO_TRADE" in swing[0]:
        lines.append(f"🚫 本日は新規見送り（{swing[0]['reason']}）")
    elif swing:
        for c in swing:
            lines.append(
                f"- {c['ticker']} [{c['sector']}]\n"
                f"  形:{c['setup']}  RR:{c['R']:.2f}  EV:{c['ev']:.2f}\n"
                f"  IN:{c['in_low']:.1f}〜{c['in_high']:.1f}\n"
                f"  STOP:{c['stop']:.1f}  TP1:{c['tp1']:.1f}  TP2:{c['tp2']:.1f}\n"
                f"  行動:{c['action']}\n"
            )
    else:
        lines.append("- 該当なし")

    lines.append("📊 ポジション")
    lines.append(pos_text or "ノーポジション")

    return "\n".join(lines)

# ============================================================
# LINE送信（既存仕様）
# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return

    for i in range(0, len(text), 3800):
        requests.post(WORKER_URL, json={"text": text[i:i+3800]}, timeout=20)

# ============================================================
# main
# ============================================================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, _ = analyze_positions(pos_df, mkt_score=mkt["score"])

    report = build_report(today_str, today_date, mkt, pos_text)
    print(report)
    send_line(report)

if __name__ == "__main__":
    main()