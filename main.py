from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.rr import compute_rr
from utils.util import jst_today_str


# ============================================================
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SCREENING_TOP_N = 10
MAX_FINAL_STOCKS = 3 

# earnings filter
EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# JST helper
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ============================================================
# events
# ============================================================
def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    events = []
    for _, r in df.iterrows():
        date_str = str(r.get("date", "")).strip()
        label = str(r.get("label", "")).strip()
        kind = str(r.get("kind", "")).strip()
        if not date_str or not label:
            continue
        events.append({"date": date_str, "label": label, "kind": kind})
    return events


def build_event_warnings(today: datetime.date) -> List[str]:
    evs = load_events()
    warns = []
    for ev in evs:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        delta = (d - today).days
        if -1 <= delta <= 2:
            if delta > 0:
                when = f"{delta}日後"
            elif delta == 0:
                when = "本日"
            else:
                when = "直近"
            warns.append(f"⚠ {ev['label']}（{when}）")
    return warns


# ============================================================
# Universe
# ============================================================
def load_universe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "ticker" not in df.columns:
        return None
    df["ticker"] = df["ticker"].astype(str)

    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT
    return df


def in_earnings_window(row: pd.Series, today: datetime.date) -> bool:
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        return abs((d - today).days) <= EARNINGS_EXCLUDE_DAYS
    except Exception:
        return False


# ============================================================
# yfinance
# ============================================================
def fetch_history(ticker: str, period: str = "120d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(1.0)
    return None


# ============================================================
# Score → RR screening
# ============================================================
def run_screening(today: datetime.date, mkt_score: int) -> List[Dict]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

    cands = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue
        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        base_score = score_stock(hist)
        if base_score is None or not np.isfinite(base_score):
            continue

        rr_info = compute_rr(hist, mkt_score)
        rr = float(rr_info.get("rr", 0.0))
        tp_pct = float(rr_info.get("tp_pct", 0.0))
        sl_pct = float(rr_info.get("sl_pct", 0.0))
        entry = float(rr_info.get("entry", 0.0))

        if rr < 1.5:
            continue

        cands.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": base_score,
            "rr": rr,
            "entry": entry,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
        })

    cands.sort(key=lambda x: x["rr"], reverse=True)
    return cands[:MAX_FINAL_STOCKS]


# ============================================================
# Report
# ============================================================
def build_report(today_str: str, today_date: datetime.date,
                 mkt: Dict, total_asset: float, pos_text: str) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    est_asset = total_asset if np.isfinite(total_asset) and total_asset > 0 else 2_000_000.0
    est_asset_int = int(round(est_asset))
    max_pos = int(round(est_asset * 1.3))

    secs = top_sectors_5d()
    sec_text = "\n".join(
        f"{i+1}. {name} (+{chg:.2f}%)" for i, (name, chg) in enumerate(secs[:5])
    )

    events = build_event_warnings(today_date)
    if not events:
        events = ["特になし"]

    core = run_screening(today_date, mkt_score)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- レバ: 1.3倍（中立（押し目））")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")
    lines.append("📈 セクター（5日）")
    lines.append(sec_text)
    lines.append("")
    lines.append("⚠ イベント")
    for ev in events:
        lines.append(f"⚠ {ev}")
    lines.append("")
    lines.append(f"🏆 Core候補（最大{MAX_FINAL_STOCKS}銘柄）")

    for c in core:
        lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
        lines.append(f"Score:{c['score']:.1f} RR:{c['rr']:.2f}R")
        lines.append(f"IN:{c['entry']:.1f} TP:{c['tp_pct']*100:.1f}% SL:{c['sl_pct']*100:.1f}%")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip())

    return "\n".join(lines)


# ============================================================
# send LINE
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print(text)
        return
    chunks = [text[i:i+3900] for i in range(0, len(text), 3900)] or [""]
    for ch in chunks:
        try:
            requests.post(WORKER_URL, json={"text": ch}, timeout=15)
        except Exception:
            pass


# ============================================================
# Entry
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today = jst_today_date()

    mkt = enhance_market_score()

    # ★ 修正済み：戻り値5つを受け取る
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk_info = analyze_positions(pos_df)

    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0

    report = build_report(today_str, today, mkt, total_asset, pos_text)

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()