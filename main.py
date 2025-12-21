from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import score_stock, calc_inout_for_stock, trend_gate
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions

# ============================================================
# 設定（Swing専用）
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SWING_MAX_FINAL = 5
SCORE_MIN = 80.0
RR_MIN = 2.0
EV_MIN = 0.40
SECTOR_TOP_N = 5

# ============================================================
def expected_r(in_rank: str, rr: float) -> float:
    win = {"強IN": 0.45, "通常IN": 0.40}.get(in_rank, 0.25)
    return win * rr - (1 - win)

def fetch_history(ticker: str):
    try:
        df = yf.Ticker(ticker).history(period="260d", auto_adjust=True)
        return df if df is not None and len(df) >= 120 else None
    except Exception:
        return None

# ============================================================
def run_swing(today_date, mkt_score: int):
    uni = pd.read_csv(UNIVERSE_PATH)
    t_col = "ticker" if "ticker" in uni.columns else "code"

    out = []
    for _, r in uni.iterrows():
        ticker = str(r[t_col]).strip()
        hist = fetch_history(ticker)
        if hist is None:
            continue

        if not trend_gate(hist):
            continue

        score = score_stock(hist)
        if score is None or score < SCORE_MIN:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score)
        rr = rr_info["rr"]
        if rr < RR_MIN:
            continue

        ev = expected_r(in_rank, rr)
        if ev < EV_MIN:
            continue

        price = float(hist["Close"].iloc[-1])
        entry = rr_info["entry"]

        out.append(dict(
            ticker=ticker,
            name=r.get("name", ticker),
            sector=r.get("sector", "不明"),
            rr=rr,
            ev=ev,
            in_rank=in_rank,
            entry=entry,
            price=price,
            gap=(price/entry-1)*100,
            tp=rr_info["tp_price"],
            sl=rr_info["sl_price"],
        ))

    # ★ 並び順は EV → RR のみ
    out.sort(key=lambda x: (x["ev"], x["rr"]), reverse=True)
    return out[:SWING_MAX_FINAL]

# ============================================================
def build_report(today_str, today_date, mkt, pos_text, total_asset):
    mkt_score = mkt["score"]
    lev = 1.7 if mkt_score >= 50 else 1.3
    max_pos = int(total_asset * lev)

    sectors = top_sectors_5d(SECTOR_TOP_N)
    swing = run_swing(today_date, mkt_score)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論（Swing専用）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt['comment']})")
    lines.append(f"- レバ: {lev:.1f}倍")
    lines.append(f"- MAX建玉: 約{max_pos:,}円\n")

    lines.append("📈 セクター（5日）")
    for i,(s,p) in enumerate(sectors,1):
        lines.append(f"{i}. {s} ({p:+.2f}%)")
    lines.append("")

    lines.append("🏆 Swing（順張りのみ）")
    if swing:
        evs=[x["ev"] for x in swing]
        rrs=[x["rr"] for x in swing]
        lines.append(f"  候補数:{len(swing)}銘柄 / 平均RR:{np.mean(rrs):.2f} / 平均EV:{np.mean(evs):.2f}\n")
        for c in swing:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  RR:{c['rr']:.2f} EV:{c['ev']:.2f} IN:{c['in_rank']}")
            lines.append(f"  IN:{c['entry']:.1f} 現在:{c['price']:.1f} ({c['gap']:+.2f}%)")
            lines.append(f"  TP:{c['tp']:.1f} SL:{c['sl']:.1f}\n")
    else:
        lines.append("- 該当なし\n")

    lines.append("📊 ポジション")
    lines.append(pos_text)
    return "\n".join(lines)

# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return
    for ch in [text[i:i+3800] for i in range(0,len(text),3800)]:
        requests.post(WORKER_URL, json={"text": ch}, timeout=20)

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