from __future__ import annotations
import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.util import jst_today_str
from utils.rr import compute_tp_sl_rr


# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SCREENING_TOP_N = 10
MAX_FINAL_STOCKS = 3
EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# JST
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ============================================================
# イベント
# ============================================================
def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    events = []
    for _, row in df.iterrows():
        date_str = str(row.get("date", "")).strip()
        label = str(row.get("label", "")).strip()
        if not date_str or not label:
            continue
        events.append({"date": date_str, "label": label})
    return events


def build_event_warnings(today: datetime.date) -> List[str]:
    events = load_events()
    warns = []
    for ev in events:
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
            warns.append(f"⚠ {ev['label']}（{when}）: サイズ注意")
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
    return abs((d - today).days) <= EARNINGS_EXCLUDE_DAYS


def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(1)
    return None


# ============================================================
# 地合い
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 1.8, "強気（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return 1.5, "やや強気（押し目）"
    if mkt_score >= 50:
        return 1.3, "中立（押し目のみ）"
    if mkt_score >= 40:
        return 1.1, "弱含み（ロット控えめ）"
    return 1.0, "守り"


def dynamic_min_score(mkt_score: int) -> float:
    if mkt_score >= 70:
        return 72
    if mkt_score >= 60:
        return 75
    if mkt_score >= 50:
        return 78
    if mkt_score >= 40:
        return 80
    return 82


def calc_max_position(total_asset: float, lev: float) -> int:
    return int(round(total_asset * lev))


# ============================================================
# スクリーニング
# ============================================================
def run_screening(today: datetime.date, mkt_score: int) -> List[Dict]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

    min_score = dynamic_min_score(mkt_score)

    raw = []
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        base_score = score_stock(hist)
        if base_score is None or base_score < min_score:
            continue

        # RR計算
        tp_pct, sl_pct, rr_val = compute_tp_sl_rr(hist, mkt_score)
        if rr_val < 1.5:
            continue

        close = float(hist["Close"].iloc[-1])
        entry = close * 0.995  # 簡易: 現値近辺

        raw.append({
            "ticker": ticker,
            "name": name,
            "score": base_score,
            "price": close,
            "entry": entry,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "rr": rr_val,
        })

    raw.sort(key=lambda x: x["score"], reverse=True)
    return raw[:MAX_FINAL_STOCKS]


# ============================================================
# レポート
# ============================================================
def build_report(today_str: str, today_date: datetime.date,
                 mkt_score: int, mkt_comment: str,
                 total_asset: float, pos_text: str) -> str:

    rec_lev, lev_comment = recommend_leverage(mkt_score)
    est = int(round(total_asset))
    max_pos = calc_max_position(total_asset, rec_lev)
    warns = build_event_warnings(today_date)

    core = run_screening(today_date, mkt_score)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点（{mkt_comment}）")
    lines.append(f"- レバ: {rec_lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    lines.append("⚠ イベント")
    if not warns:
        lines.append("- 特筆すべきイベントなし")
    else:
        for ev in warns:
            lines.append(ev)
    lines.append("")

    lines.append(f"🏆 Core候補（最大{MAX_FINAL_STOCKS}）")
    if not core:
        lines.append("該当なし")
    else:
        for c in core:
            lines.append(f"- {c['ticker']} {c['name']}")
            lines.append(f"Score:{c['score']:.1f} RR:{c['rr']:.2f}R")
            lines.append(f"IN:{c['entry']:.1f} TP:+{c['tp_pct']*100:.1f}% SL:{c['sl_pct']*100:.1f}%")
            lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip())

    return "\n".join(lines)


# ============================================================
# LINE send
# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return
    chunks = [text[i:i+3800] for i in range(0, len(text), 3800)]
    for ch in chunks:
        try:
            requests.post(WORKER_URL, json={"text": ch}, timeout=15)
        except Exception:
            print(ch)


# ============================================================
# main
# ============================================================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    m = calc_market_score()
    mkt_score = int(m.get("score", 50))
    mkt_comment = str(m.get("comment", ""))

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, *_ = analyze_positions(pos_df)
    if not np.isfinite(total_asset) or total_asset <= 0:
        total_asset = 2_000_000.0

    report = build_report(today_str, today_date,
                          mkt_score, mkt_comment,
                          total_asset, pos_text)

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()