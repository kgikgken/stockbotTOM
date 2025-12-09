from __future__ import annotations

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.rr import compute_rr

# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")
EARNINGS_EXCLUDE_DAYS = 3
MAX_FINAL_STOCKS = 3


# ============================================================
# 日付系
# ============================================================
def jst_now() -> datetime:
    return datetime.now().astimezone(timezone(timedelta(hours=9)))

def jst_str_now() -> str:
    return jst_now().strftime("%Y-%m-%d")


# ============================================================
# Earnings 除外
# ============================================================
def filter_earnings(df: pd.DataFrame, today: datetime) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df

    try:
        df["earnings_date"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.tz_localize("Asia/Tokyo", nonexistent="shift_forward", ambiguous="NaT")
    except Exception:
        return df

    today = today.astimezone(timezone(timedelta(hours=9)))
    delta = (df["earnings_date"] - today).dt.days.abs()
    return df[delta > EARNINGS_EXCLUDE_DAYS]


# ============================================================
# スクリーニング
# ============================================================
def run_screening(today: datetime, mkt_score: int) -> List[Dict]:
    df = pd.read_csv(UNIVERSE_PATH)
    df = filter_earnings(df, today)

    results = []
    for _, row in df.iterrows():
        ticker = row["code"]
        try:
            hist = yf.download(
                ticker,
                period="50d",
                interval="1d",
                auto_adjust=True,
                progress=False
            )
            if hist is None or len(hist) < 20:
                continue

            score = float(row.get("score", 0))
            rr_info = compute_rr(hist, mkt_score)
            rr = float(rr_info["rr"])

            results.append(dict(
                ticker=ticker,
                sector=row.get("sector", ""),
                score=score,
                rr=rr,
                entry=rr_info["entry"],
                tp_pct=rr_info["tp_pct"],
                sl_pct=rr_info["sl_pct"],
            ))
        except Exception:
            continue

    results = sorted(results, key=lambda x: (x["score"], x["rr"]), reverse=True)
    results = [r for r in results if r["rr"] >= 1.5]
    return results[:MAX_FINAL_STOCKS]


# ============================================================
# レポート作成
# ============================================================
def build_report(today_str: str, today: datetime, mkt_score: int,
                 pos_text: str, total_asset: float) -> str:

    core = run_screening(today, mkt_score)

    sect = top_sectors_5d()

    text = []
    text.append(f"📅 {today_str} stockbotTOM 日報\n")
    text.append("◆ 今日の結論")
    text.append(f"- 地合い: {mkt_score}点 (中立)")
    lever = round(1.0 + (mkt_score - 50) / 100, 1)
    text.append(f"- レバ: {lever}倍（中立（押し目））")
    text.append(f"- MAX建玉: 約{int(total_asset * lever):,}円\n")

    text.append("📈 セクター（5日）")
    for i, s in enumerate(sect):
        if i >= 5:
            break
        text.append(f"{i+1}. {s['sector']} ({s['pct']:+.2f}%)")
    text.append("")

    text.append("🏆 Core候補（最大3銘柄）")
    if core:
        for r in core:
            text.append(f"- {r['ticker']} [{r['sector']}]")
            text.append(f"Score:{r['score']:.1f} RR:{r['rr']:.2f}R")
            text.append(f"IN:{r['entry']:.1f} TP:{r['tp_pct']*100:+.1f}% SL:{r['sl_pct']*100:.1f}%\n")
    else:
        text.append("- 該当なし\n")

    text.append("\n📊 ポジション")
    text.append(pos_text)
    return "\n".join(text)


# ============================================================
# LINE送信
# ============================================================
def send_line(msg: str):
    if not WORKER_URL:
        print(msg)
        return
    import requests
    try:
        requests.post(WORKER_URL, json={"message": msg}, timeout=10)
    except Exception:
        print(msg)


# ============================================================
# Main
# ============================================================
def main():
    today = jst_now()
    today_str = jst_str_now()

    mkt_score = calc_market_score()
    pos_df = load_positions()
    pos_text, total_asset = analyze_positions(pos_df)

    report = build_report(today_str, today, mkt_score, pos_text, total_asset)
    send_line(report)


if __name__ == "__main__":
    main()