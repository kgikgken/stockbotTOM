from __future__ import annotations

import os
import time
from typing import List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import score_stock, trend_gate, strong_in_only
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions

UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SCORE_MIN = 75.0
RR_MIN = 2.5
RR_MAX = 4.5
EV_MIN = 1.0
MAX_GAP_PCT = 1.0   # 遅い銘柄排除

# -------------------------
def fetch_history(ticker: str, period="300d"):
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        return df if df is not None and len(df) > 120 else None
    except Exception:
        return None

# -------------------------
def expected_r(rr: float, win_rate: float = 0.45) -> float:
    return win_rate * rr - (1 - win_rate)

# -------------------------
def load_events():
    if not os.path.exists(EVENTS_PATH):
        return []
    df = pd.read_csv(EVENTS_PATH)
    return df.to_dict("records")

# -------------------------
def run_swing(today_date) -> List[Dict]:
    uni = pd.read_csv(UNIVERSE_PATH)

    t_col = "ticker" if "ticker" in uni.columns else "code"
    out = []

    for _, row in uni.iterrows():
        ticker = str(row[t_col])
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        hist = fetch_history(ticker)
        if hist is None:
            continue

        # ① トレンドゲート
        if not trend_gate(hist):
            continue

        # ② スコア
        score = score_stock(hist)
        if score is None or score < SCORE_MIN:
            continue

        # ③ 強INのみ
        in_rank, entry = strong_in_only(hist)
        if in_rank != "強IN":
            continue

        # ④ RR
        rr_info = compute_tp_sl_rr(hist, mkt_score=50)
        rr = rr_info["rr"]
        if rr < RR_MIN or rr > RR_MAX:
            continue

        # ⑤ EV
        ev = expected_r(rr)
        if ev < EV_MIN:
            continue

        price_now = float(hist["Close"].iloc[-1])
        gap_pct = (price_now / entry - 1) * 100
        if gap_pct > MAX_GAP_PCT:
            continue

        out.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": score,
            "rr": rr,
            "ev": ev,
            "entry": entry,
            "price_now": price_now,
            "gap_pct": gap_pct,
            "tp": rr_info["tp_price"],
            "sl": rr_info["sl_price"],
        })

    out.sort(key=lambda x: (x["ev"], x["rr"]), reverse=True)
    return out

# -------------------------
def build_report(today_str, mkt, swing, pos_text):
    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論（Swing専用）")
    lines.append(f"- 地合い: {mkt['score']}点 ({mkt['comment']})\n")

    lines.append("📈 セクター（5日）")
    for i, (s, p) in enumerate(top_sectors_5d(), 1):
        lines.append(f"{i}. {s} ({p:+.2f}%)")
    lines.append("")

    lines.append("🏆 Swing（順張りのみ）")
    if not swing:
        lines.append("- 該当なし（勝てる形なし）\n")
    else:
        for c in swing:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(
                f"  Score:{c['score']:.1f} RR:{c['rr']:.2f} EV:{c['ev']:.2f} IN:強IN"
            )
            lines.append(
                f"  IN:{c['entry']:.1f} 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)"
            )
            lines.append(
                f"  TP:{c['tp']:.1f} SL:{c['sl']:.1f}\n"
            )

    lines.append("📊 ポジション")
    lines.append(pos_text)

    return "\n".join(lines)

# -------------------------
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return
    requests.post(WORKER_URL, json={"text": text}, timeout=20)

# -------------------------
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, _ = analyze_positions(pos_df, mkt_score=mkt["score"])

    swing = run_swing(today_date)
    report = build_report(today_str, mkt, swing, pos_text)

    print(report)
    send_line(report)

if __name__ == "__main__":
    main()