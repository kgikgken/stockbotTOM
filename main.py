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
from utils.rr import compute_tp_sl_rr


# ============================================================
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"      # 任意
WORKER_URL = os.getenv("WORKER_URL")

SCREENING_TOP_N = 15
MAX_FINAL_STOCKS = 5
EARNINGS_EXCLUDE_DAYS = 3

MAX_CORE_POSITIONS = 3
RISK_PER_TRADE = 0.015
LIQ_MIN_TURNOVER = 100_000_000


# ============================================================
# ユーティリティ
# ============================================================
def jst_today_str() -> str:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).strftime("%Y-%m-%d")


def load_universe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_events(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def yf_price(ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
    try:
        hist = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
        if hist is None or hist.empty:
            return None
        return hist
    except Exception:
        return None


# ============================================================
# フィルタ
# ============================================================
def filter_earnings(df: pd.DataFrame, today: datetime) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df
    try:
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    except Exception:
        return df

    delta = (df["earnings_date"] - today).dt.days.abs()
    return df[delta > EARNINGS_EXCLUDE_DAYS]


# ============================================================
# スクリーニング
# ============================================================
def run_screening(today: datetime, mkt_score: float) -> List[Dict]:
    uni = load_universe(UNIVERSE_PATH)

    tday = pd.Timestamp(today)
    uni = filter_earnings(uni, tday)

    results = []
    for _, row in uni.iterrows():
        t = row["ticker"]
        hist = yf_price(t, 60)
        if hist is None:
            continue

        # RR
        rr_info = compute_tp_sl_rr(hist, mkt_score)
        rr = float(rr_info.get("rr", 0.0))

        if rr < 1.5:
            continue

        # Score
        sc = score_stock(t, hist)

        results.append(dict(
            ticker=t,
            score=float(sc),
            rr=float(rr),
            entry=rr_info.get("entry", 0.0),
            tp_pct=rr_info.get("tp_pct", 0.0),
            sl_pct=rr_info.get("sl_pct", 0.0),
            sector=row.get("sector", ""),
        ))

    df = pd.DataFrame(results)
    if df.empty:
        return []

    df = df.sort_values("score", ascending=False).head(SCREENING_TOP_N)
    top = df.sort_values(["score", "rr"], ascending=False).head(MAX_FINAL_STOCKS)
    return top.to_dict(orient="records")


# ============================================================
# レポート
# ============================================================
def load_pos_text_and_asset() -> Tuple[str, float]:
    if not os.path.exists(POSITIONS_PATH):
        return "", 0.0

    df = pd.read_csv(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(df)
    return pos_text, total_asset


def build_report(today_str: str, today: datetime, mkt_score: float,
                 pos_text: str, total_asset: float) -> str:

    core_list = run_screening(today, mkt_score)
    t5 = top_sectors_5d()
    ev = load_events(EVENTS_PATH)

    txt = []
    txt.append(f"📅 {today_str} stockbotTOM 日報\n")
    txt.append("◆ 今日の結論")
    txt.append(f"- 地合い: {int(mkt_score)}点 (中立)")
    lev = "1.3倍（中立（押し目））"
    txt.append(f"- レバ: {lev}")
    txt.append(f"- MAX建玉: 約{int(total_asset * 1.3):,}円\n")

    if t5 is not None and not t5.empty:
        txt.append("📈 セクター（5日）")
        for i, row in t5.head(5).iterrows():
            txt.append(f"{i+1}. {row['sector']} ({row['ret_pct']:+.2f}%)")
        txt.append("")

    if ev is not None and not ev.empty:
        coming = ev[ev["days_to"] >= 0].sort_values("days_to").head(3)
        for _, r in coming.iterrows():
            txt.append(f"⚠ {r['event']}（{r['days_to']}日後）")
        txt.append("")

    txt.append("🏆 Core候補（最大3銘柄）")
    for r in core_list[:3]:
        txt.append(f"- {r['ticker']} {r['sector']}")
        txt.append(f"Score:{r['score']:.1f} RR:{r['rr']:.2f}R")
        txt.append(f"IN:{r['entry']:.1f} TP:{r['tp_pct']*100:+.1f}% SL:{r['sl_pct']*100:+.1f}%\n")

    txt.append("📊 ポジション")
    if pos_text:
        txt.append(pos_text)
    else:
        txt.append("ノーポジション")

    return "\n".join(txt)


# ============================================================
# LINE Push
# ============================================================
def push_line(msg: str):
    if not WORKER_URL:
        return
    try:
        requests.post(WORKER_URL, json={"text": msg}, timeout=10)
    except Exception:
        pass


# ============================================================
# main
# ============================================================
def main():
    today = datetime.now(timezone(timedelta(hours=9)))
    today_str = today.strftime("%Y-%m-%d")

    hist = yf_price("^TOPX", 50)
    if hist is None:
        mkt = dict(score=50)
    else:
        mkt = calc_market_score(hist)

    mkt_score = float(mkt.get("score", 50))

    pos_text, total_asset = load_pos_text_and_asset()
    report = build_report(today_str, today, mkt_score, pos_text, total_asset)
    push_line(report)


if __name__ == "__main__":
    main()