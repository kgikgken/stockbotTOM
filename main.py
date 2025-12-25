from __future__ import annotations

import os
import time
from typing import List, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import judge_setup_and_pwin
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions

# ============================================================
# 設定（Swing専用）
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

MAX_FINAL = 5
EARNINGS_EXCLUDE_DAYS = 3
ATR_PCT_MIN = 0.015
ADV_MIN = 100_000_000

# ============================================================
# データ取得
# ============================================================
def fetch_history(ticker: str, period: str = "260d"):
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and len(df) >= 120:
                return df
        except Exception:
            time.sleep(0.3)
    return None

# ============================================================
# 決算除外
# ============================================================
def filter_earnings(df: pd.DataFrame, today):
    if "earnings_date" not in df.columns:
        return df
    d = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    keep = []
    for x in d:
        if pd.isna(x):
            keep.append(True)
        else:
            keep.append(abs((x - today).days) > EARNINGS_EXCLUDE_DAYS)
    return df[keep]

# ============================================================
# EV
# ============================================================
def calc_ev(pwin: float, rr: float) -> float:
    return pwin * rr - (1 - pwin) * 1.0

# ============================================================
# Swing Screener
# ============================================================
def run_swing(today, mkt_score: int) -> List[Dict]:
    uni = pd.read_csv(UNIVERSE_PATH)
    t_col = "ticker" if "ticker" in uni.columns else "code"
    uni = filter_earnings(uni, today)

    sectors_top = [s for s, _ in top_sectors_5d(5)]
    out = []

    for _, row in uni.iterrows():
        ticker = str(row.get(t_col, "")).strip()
        if not ticker:
            continue

        sector = str(row.get("sector", row.get("industry_big", "")))
        if sector not in sectors_top:
            continue

        hist = fetch_history(ticker)
        if hist is None:
            continue

        close = hist["Close"].iloc[-1]
        atr = hist["High"].rolling(14).max().iloc[-1] - hist["Low"].rolling(14).min().iloc[-1]
        if atr / close < ATR_PCT_MIN:
            continue

        turnover = (hist["Close"] * hist["Volume"]).rolling(20).mean().iloc[-1]
        if not np.isfinite(turnover) or turnover < ADV_MIN:
            continue

        setup, pwin = judge_setup_and_pwin(hist)
        if setup not in ("A", "B"):
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score)
        rr = rr_info["rr"]
        if rr < 2.2:
            continue

        ev = calc_ev(pwin, rr)
        if ev < 0.4:
            continue

        expected_days = max(1.0, (rr_info["tp_price"] - rr_info["entry"]) / max(rr_info["entry"] * 0.01, 1))
        r_day = rr / expected_days
        if expected_days > 5 or r_day < 0.5:
            continue

        price_now = hist["Close"].iloc[-1]
        gap = (price_now - rr_info["entry"]) / rr_info["entry"]

        if gap > 0.8 * (atr / rr_info["entry"]):
            action = "WATCH_ONLY"
        elif gap > 0:
            action = "LIMIT_WAIT"
        else:
            action = "EXEC_NOW"

        out.append({
            "ticker": ticker,
            "sector": sector,
            "setup": setup,
            "rr": rr,
            "ev": ev,
            "r_day": r_day,
            "entry": rr_info["entry"],
            "price": price_now,
            "tp1": rr_info["entry"] + (rr_info["tp_price"] - rr_info["entry"]) * 0.5,
            "tp2": rr_info["tp_price"],
            "sl": rr_info["sl_price"],
            "expected_days": expected_days,
            "action": action,
        })

    out.sort(key=lambda x: (x["ev"], x["r_day"], x["rr"]), reverse=True)
    return out[:MAX_FINAL]

# ============================================================
# レポート
# ============================================================
def build_report(today_str, today_date, mkt, pos_text, total_asset):
    swing = run_swing(today_date, mkt["score"])
    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用）")
    lines.append("✅ 新規可（条件クリア）")
    lines.append(f"- 地合い: {mkt['score']}点 ({mkt['comment']})")
    lines.append("")
    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止）")

    if not swing:
        lines.append("- 該当なし")
    else:
        for c in swing:
            lines.append(f"- {c['ticker']} [{c['sector']}] ⭐")
            lines.append(
                f"  形:{c['setup']} RR:{c['rr']:.2f} EV:{c['ev']:.2f} R/day:{c['r_day']:.2f}"
            )
            lines.append(
                f"  IN:{c['entry']:.1f} 現在:{c['price']:.1f} 行動:{c['action']}"
            )
            lines.append(
                f"  TP1:{c['tp1']:.1f} TP2:{c['tp2']:.1f} SL:{c['sl']:.1f}"
            )
            lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text)

    return "\n".join(lines)

# ============================================================
# LINE送信（届く仕様）
# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return
    for i in range(0, len(text), 3800):
        requests.post(WORKER_URL, json={"text": text[i:i+3800]}, timeout=20)

# ============================================================
# Main
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
    main()