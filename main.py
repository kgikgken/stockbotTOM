from __future__ import annotations
import os
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.rr import compute_tp_sl_rr

# =========================
# 基本設定
# =========================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# =========================
#	Send LINE
# =========================
def send_line(text: str):
    try:
        payload = {"text": text}
        r = requests.post(WORKER_URL, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception:
        return False

# =========================
#	日付(JST)
# =========================
def jst_today():
    JST = timezone(timedelta(hours=9))
    return datetime.now(JST)

def jst_today_str():
    return jst_today().strftime("%Y-%m-%d")

# =========================
#	ポジション読み込み
# =========================
def load_pos_text_and_asset():
    df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(df)
    return pos_text, total_asset

# =========================
#	スクリーニング（Score + RR）
# =========================
def run_screening(today: datetime, mkt_score: float):
    uni = pd.read_csv(UNIVERSE_PATH)
    tickers = uni["ticker"].tolist()

    results = []

    for t in tickers:
        try:
            hist = yf.download(t, period="50d", interval="1d", progress=False)
            if hist is None or hist.empty:
                continue
        except Exception:
            continue

        stock_score = score_stock(t, hist, uni)

        rr_info = compute_tp_sl_rr(hist, mkt_score)
        rr_val = float(rr_info.get("rr", 0.0))
        tp_pct = rr_info.get("tp_pct", 0.0)
        sl_pct = rr_info.get("sl_pct", 0.0)

        # --- フィルタ ---
        if rr_val < 2.0:
            continue
        if stock_score < 70:
            continue

        # --- 複合評価 ---
        final_score = stock_score * 0.5 + rr_val * 0.5

        results.append({
            "ticker": t,
            "score": stock_score,
            "rr": rr_val,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "final": final_score
        })

    results = sorted(results, key=lambda x: x["final"], reverse=True)

    return results[:3]

# =========================
#	セクター表示
# =========================
def build_sector_text(today: datetime):
    sect = top_sectors_5d()
    if not sect:
        return ""
    lines = []
    lines.append("📈 セクター（5日）")
    for i, (name, pct) in enumerate(sect[:5], start=1):
        lines.append(f"{i}. {name} ({pct:+.2f}%)")
    return "\n".join(lines)

# =========================
#	レポート生成
# =========================
def build_report(today_str, today, mkt_score, pos_text, total_asset):
    core = run_screening(today, mkt_score)
    sector_txt = build_sector_text(today)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 (中立)")
    lines.append(f"- レバ: 1.3倍（中立（押し目））")
    lines.append(f"- MAX建玉: 約{int(total_asset*1.3):,}円\n")

    if sector_txt:
        lines.append(sector_txt + "\n")

    # ----- Core -----
    lines.append("🏆 Core候補（最大3銘柄）")
    for c in core:
        ticker = c["ticker"]
        score = round(c["score"], 1)
        rr = round(c["rr"], 2)
        tp = round(c["tp_pct"] * 100, 1)
        sl = round(c["sl_pct"] * 100, 1)

        lines.append(f"- {ticker} Score:{score} RR:{rr}R")
        lines.append(f"IN:??? TP:{tp}% SL:{sl}%\n")

    # ----- ポジション -----
    lines.append("📊 ポジション")
    lines.append(pos_text)

    return "\n".join(lines)

# =========================
#	イベント警告
# =========================
def find_near_event(today: datetime):
    if not os.path.exists(EVENTS_PATH):
        return None
    df = pd.read_csv(EVENTS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    for _, row in df.iterrows():
        d = row["date"].date()
        if 0 <= (d - today.date()).days <= 3:
            return f"⚠ {row['label']}（{(d - today.date()).days}日後）"
    return None

# =========================
#	Main
# =========================
def main():
    today = jst_today()
    today_str = jst_today_str()

    pos_text, total_asset = load_pos_text_and_asset()
    mkt_score = calc_market_score()

    event = find_near_event(today)

    report = build_report(today_str, today, mkt_score, pos_text, total_asset)

    if event:
        report = report.replace("🏆", f"⚠ イベント\n{event}\n\n🏆")

    send_line(report)


if __name__ == "__main__":
    main()