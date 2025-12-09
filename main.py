from __future__ import annotations
import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.scoring import score_stock
from utils.position import load_positions, analyze_positions
from utils.rr import compute_tp_sl_rr
from utils.util import jst_today_str


UNIVERSE_PATH = "universe_jpx.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")


# ============================================================
# helpers
# ============================================================

def fetch_hist(ticker: str, days: int = 120) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=f"{days}d")
        if df is None or len(df) == 0:
            return None
        return df
    except Exception:
        return None


# ============================================================
# screening core
# ============================================================

def run_screening(today: datetime, mkt_score: int):

    df = pd.read_csv(UNIVERSE_PATH)
    results = []

    for _, row in df.iterrows():
        ticker = row["symbol"]

        hist = fetch_hist(ticker, 120)
        if hist is None or len(hist) < 60:
            continue

        score = score_stock(hist)
        if score is None or score < 70:
            continue

        tp, sl, rr = compute_tp_sl_rr(hist, mkt_score)
        if rr < 1.5:
            continue

        last = float(hist["Close"].iloc[-1])
        in_price = last * 0.99

        results.append({
            "ticker": ticker,
            "score": round(score, 1),
            "rr": round(rr, 2),
            "in": round(in_price, 1),
            "tp": round(tp, 1),
            "sl": round(sl, 1),
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:3]


# ============================================================
# report
# ============================================================

def build_report(today_str: str, today: datetime, mkt_score: int, pos_text: str):

    txt = []
    txt.append(f"📅 {today_str} stockbotTOM 日報\n")
    txt.append("◆ 今日の結論")
    txt.append(f"- 地合い: {mkt_score}点 (中立)")
    txt.append("- レバ: 1.3倍（中立（押し目））")
    txt.append("- MAX建玉: 約2,600,000円\n")

    # sector
    sectors = top_sectors_5d()
    txt.append("📈 セクター（5日）")
    for name, pct in sectors[:5]:
        txt.append(f"{name} ({pct:+.2f}%)")
    txt.append("")

    # events
    txt.append("⚠ イベント")
    txt.append("⚠ FOMC（2日後）\n")

    # core
    txt.append("🏆 Core候補（最大3銘柄）")
    for r in run_screening(today, mkt_score):
        txt.append(
            f"- {r['ticker']}  Score:{r['score']} RR:{r['rr']}R\n"
            f"IN:{r['in']} TP:{r['tp']}% SL:{r['sl']}%\n"
        )

    txt.append("🔄 ポジション入れ替え候補（RR差ベース）")
    txt.append("- 明確な入れ替え候補なし\n")

    txt.append("📊 ポジション")
    txt.append(pos_text)

    return "\n".join(txt)


# ============================================================
# line
# ============================================================

def send_line(text: str):
    try:
        r = requests.post(WORKER_URL, json={"text": text}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print("LINE error:", e)
        return False


# ============================================================
# main
# ============================================================

def main():
    today = datetime.now()
    today_str = jst_today_str()

    mkt_score = calc_market_score()
    pos_df = load_positions()
    pos_text, total_asset = analyze_positions(pos_df)

    report = build_report(today_str, today, mkt_score, pos_text)
    send_line(report)


if __name__ == "__main__":
    main()