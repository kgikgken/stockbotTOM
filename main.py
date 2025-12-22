from __future__ import annotations

import os
import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import (
    score_stock,
    calc_inout_for_stock,
    trend_gate
)
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions

# ============================================================
# 設定（Swing専用・順張りのみ）
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

EARNINGS_EXCLUDE_DAYS = 3

SWING_MAX_FINAL = 5
SWING_SCORE_MIN = 72.0
SWING_RR_MIN = 2.0
SWING_EV_R_MIN = 0.40

SECTOR_TOP_N = 5


# ============================================================
# util
# ============================================================
def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def fetch_history(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.5)
    return None


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

    out = []
    for _, r in df.iterrows():
        label = str(r.get("label", "")).strip()
        if not label:
            continue
        out.append(
            dict(
                label=label,
                date=str(r.get("date", "")).strip(),
                time=str(r.get("time", "")).strip(),
                datetime=str(r.get("datetime", "")).strip(),
            )
        )
    return out


def build_event_warnings(today_date) -> List[str]:
    events = load_events()
    warns = []

    for ev in events:
        dt = parse_event_datetime_jst(ev["datetime"], ev["date"], ev["time"])
        if dt is None:
            continue
        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            warns.append(f"⚠ {ev['label']}（{dt.strftime('%Y-%m-%d %H:%M JST')} / {when}）")

    if not warns:
        warns.append("- 特になし")
    return warns


# ============================================================
# earnings filter
# ============================================================
def filter_earnings(df: pd.DataFrame, today_date) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df

    d = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    keep = []
    for x in d:
        if pd.isna(x):
            keep.append(True)
        else:
            keep.append(abs((x - today_date).days) > EARNINGS_EXCLUDE_DAYS)
    return df[keep]


# ============================================================
# EV
# ============================================================
def expected_r(in_rank: str, rr: float) -> float:
    win = {"強IN": 0.45, "通常IN": 0.40}.get(in_rank, 0.0)
    return win * rr - (1 - win)


# ============================================================
# Swing screening（順張りのみ）
# ============================================================
def run_swing(today_date, mkt_score: int) -> List[Dict]:
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    t_col = "ticker" if "ticker" in uni.columns else "code"
    uni = filter_earnings(uni, today_date)

    cands = []

    for _, r in uni.iterrows():
        ticker = str(r.get(t_col, "")).strip()
        if not ticker:
            continue

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 120:
            continue

        # --- トレンドゲート（逆張り完全排除） ---
        if not trend_gate(hist):
            continue

        score = score_stock(hist)
        if score is None or score < SWING_SCORE_MIN:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score)
        rr = float(rr_info["rr"])
        if rr < SWING_RR_MIN:
            continue

        ev = expected_r(in_rank, rr)
        if ev < SWING_EV_R_MIN:
            continue

        price_now = _safe_float(hist["Close"].iloc[-1])
        entry = float(rr_info["entry"])
        gap = (price_now / entry - 1) * 100 if entry > 0 else np.nan

        cands.append(
            dict(
                ticker=ticker,
                name=str(r.get("name", ticker)),
                sector=str(r.get("sector", "不明")),
                in_rank=in_rank,
                rr=rr,
                ev=ev,
                entry=entry,
                price_now=price_now,
                gap_pct=gap,
                tp_price=rr_info["tp_price"],
                sl_price=rr_info["sl_price"],
            )
        )

    cands.sort(key=lambda x: (x["ev"], x["rr"]), reverse=True)
    return cands[:SWING_MAX_FINAL]


# ============================================================
# report
# ============================================================
def build_report(today_str, today_date, mkt, pos_text, total_asset) -> str:
    mkt_score = int(mkt["score"])
    lev = 1.7 if mkt_score >= 50 else 1.3
    max_pos = int(total_asset * lev)

    sectors = top_sectors_5d(SECTOR_TOP_N)
    events = build_event_warnings(today_date)
    swing = run_swing(today_date, mkt_score)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論（Swing専用）")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt['comment']})")
    lines.append(f"- レバ: {lev:.1f}倍")
    lines.append(f"- MAX建玉: 約{max_pos:,}円\n")

    lines.append("📈 セクター（5日）")
    for i, (s, p) in enumerate(sectors, 1):
        lines.append(f"{i}. {s} ({p:+.2f}%)")
    lines.append("")

    lines.append("⚠ イベント")
    lines.extend(events)
    lines.append("")

    lines.append("🏆 Swing（順張りのみ）")
    if swing:
        avg_rr = np.mean([c["rr"] for c in swing])
        avg_ev = np.mean([c["ev"] for c in swing])
        lines.append(f"  候補数:{len(swing)}銘柄 / 平均RR:{avg_rr:.2f} / 平均EV:{avg_ev:.2f}\n")

        for c in swing:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  RR:{c['rr']:.2f} EV:{c['ev']:.2f} IN:{c['in_rank']}")
            lines.append(f"  IN:{c['entry']:.1f} 現在:{c['price_now']:.1f} ({c['gap_pct']:+.2f}%)")
            lines.append(f"  TP:{c['tp_price']:.1f} SL:{c['sl_price']:.1f}\n")
    else:
        lines.append("- 該当なし\n")

    lines.append("📊 ポジション")
    lines.append(pos_text)

    return "\n".join(lines)


# ============================================================
# LINE
# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return
    for ch in [text[i:i+3800] for i in range(0, len(text), 3800)]:
        requests.post(WORKER_URL, json={"text": ch}, timeout=20)


# ============================================================
# main
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