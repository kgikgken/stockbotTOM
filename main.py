# main.py
from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import (
    universe_filter,
    detect_setup,
    calc_entry_zone,
    calc_action,
    calc_pwin,
)
from utils.rr import calc_rr_block
from utils.position import load_positions, analyze_positions

UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

MAX_FINAL = 5
SECTOR_TOP_N = 5


def fetch_hist(ticker: str, days: int = 260) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=f"{days}d", auto_adjust=True)
        return df if df is not None and len(df) >= 200 else None
    except Exception:
        return None


def load_events() -> list[str]:
    if not os.path.exists(EVENTS_PATH):
        return ["- 特になし"]
    df = pd.read_csv(EVENTS_PATH)
    out = []
    today = jst_today_date()
    for _, r in df.iterrows():
        dt = parse_event_datetime_jst(r.get("datetime"), r.get("date"), r.get("time"))
        if dt is None:
            continue
        d = (dt.date() - today).days
        if -1 <= d <= 2:
            tag = "本日" if d == 0 else "直近" if d < 0 else f"{d}日後"
            out.append(f"⚠ {r.get('label','')}（{dt.strftime('%Y-%m-%d %H:%M JST')} / {tag}）")
    return out or ["- 特になし"]


def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return
    for i in range(0, len(text), 3800):
        requests.post(WORKER_URL, json={"text": text[i:i+3800]}, timeout=20)


def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    mkt_score = mkt["score"]
    delta3d = mkt.get("delta3d", 0)

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df, mkt_score)

    uni = pd.read_csv(UNIVERSE_PATH)
    uni = universe_filter(uni)

    sectors = top_sectors_5d(SECTOR_TOP_N)
    top_sector_names = [s[0] for s in sectors]

    cands = []
    for _, r in uni.iterrows():
        ticker = r["ticker"]
        hist = fetch_hist(ticker)
        if hist is None:
            continue

        setup = detect_setup(hist)
        if setup is None:
            continue

        if r["sector"] not in top_sector_names:
            continue

        zone = calc_entry_zone(hist, setup)
        rr_block = calc_rr_block(hist, zone, mkt_score)
        if rr_block["rr"] < 2.2:
            continue

        pwin = calc_pwin(hist, r["sector"], sectors, zone["gu"])
        ev = pwin * rr_block["rr"] - (1 - pwin)

        expected_days = rr_block["expected_days"]
        r_per_day = rr_block["rr"] / expected_days if expected_days > 0 else 0

        if ev < 0.4 or expected_days > 5 or r_per_day < 0.5:
            continue

        action = calc_action(zone)

        cands.append({
            "ticker": ticker,
            "name": r["name"],
            "sector": r["sector"],
            "setup": setup,
            "rr": rr_block["rr"],
            "ev": ev,
            "rpd": r_per_day,
            "zone": zone,
            "rr_block": rr_block,
            "action": action,
        })

    cands.sort(key=lambda x: (x["ev"], x["rpd"]), reverse=True)
    cands = cands[:MAX_FINAL]

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論（Swing専用 / 1〜7日）")
    lines.append("🚫 新規見送り" if mkt_score < 45 or (delta3d <= -5 and mkt_score < 55) else "✅ 新規可（条件クリア）")
    lines.append(f"- 地合い: {mkt_score}点")
    lines.append(f"- ΔMarketScore_3d: {delta3d}")
    lines.append("")

    lines.append("📈 セクター（5日）")
    for i, (s, p) in enumerate(sectors):
        lines.append(f"{i+1}. {s} ({p:+.2f}%)")
    lines.append("")

    lines.append("⚠ イベント")
    lines.extend(load_events())
    lines.append("")

    lines.append("🏆 Swing（順張りのみ / 追いかけ禁止 / 速度重視）")
    for c in cands:
        z = c["zone"]
        rrb = c["rr_block"]
        lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
        lines.append(f"  形:{c['setup']} RR:{c['rr']:.2f} EV:{c['ev']:.2f} R/day:{c['rpd']:.2f}")
        lines.append(f"  IN:{z['center']:.1f}（帯:{z['low']:.1f}〜{z['high']:.1f}） GU:{'Y' if z['gu'] else 'N'}")
        lines.append(f"  STOP:{rrb['stop']:.1f} TP1:{rrb['tp1']:.1f} TP2:{rrb['tp2']:.1f}")
        lines.append(f"  行動:{c['action']}")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text)

    report = "\n".join(lines)
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()