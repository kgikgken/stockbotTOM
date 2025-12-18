from __future__ import annotations

import os
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date, parse_event_datetime_jst
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.scoring import score_stock, calc_inout_for_stock
from utils.rr import compute_tp_sl_rr
from utils.position import load_positions, analyze_positions
from utils.day import score_daytrade_candidate


# ============================================================
# 設定（★LINEが届いてた時と同じ構造）
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# Swing
SWING_MAX_FINAL = 3
SWING_SCORE_MIN = 72.0
SWING_RR_MIN = 2.0
SWING_EV_R_MIN = 0.40

# Day（残すが思想はSwing集中）
DAY_MAX_FINAL = 3
DAY_SCORE_MIN = 60.0
DAY_RR_MIN = 1.5

SECTOR_TOP_N = 5
EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# 便利
# ============================================================
def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def fetch_history(ticker: str, period: str = "260d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.4)
    return None


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

    out = []
    for _, r in df.iterrows():
        label = str(r.get("label", "")).strip()
        if not label:
            continue
        out.append({
            "label": label,
            "date": str(r.get("date", "")).strip(),
            "time": str(r.get("time", "")).strip(),
            "datetime": str(r.get("datetime", "")).strip(),
        })
    return out


def build_event_warnings(today_date) -> List[str]:
    warns = []
    for ev in load_events():
        dt = parse_event_datetime_jst(ev["datetime"], ev["date"], ev["time"])
        if not dt:
            continue
        d = dt.date()
        delta = (d - today_date).days
        if -1 <= delta <= 2:
            when = "本日" if delta == 0 else ("直近" if delta < 0 else f"{delta}日後")
            warns.append(f"⚠ {ev['label']}（{dt.strftime('%Y-%m-%d %H:%M JST')} / {when}）")
    return warns or ["- 特になし"]


# ============================================================
# EV
# ============================================================
def expected_r_from_in_rank(in_rank: str, rr: float) -> float:
    if rr <= 0:
        return -999.0
    win = {"強IN": 0.45, "通常IN": 0.40, "弱めIN": 0.33}.get(in_rank, 0.25)
    return win * rr - (1 - win)


# ============================================================
# Swing（★順張り専用・逆張り完全排除）
# ============================================================
def run_swing(today_date, mkt_score: int) -> List[Dict]:
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    t_col = "ticker" if "ticker" in uni.columns else "code"
    out: List[Dict] = []

    for _, r in uni.iterrows():
        ticker = str(r.get(t_col, "")).strip()
        if not ticker:
            continue

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 120:
            continue

        close = hist["Close"]

        # ===== Trend Gate（核心）=====
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma20_prev = close.rolling(20).mean().iloc[-6]

        # 順張り条件
        if not (close.iloc[-1] > ma20 > ma50):
            continue
        if not (ma20 > ma20_prev):
            continue

        score = score_stock(hist)
        if not score or score < SWING_SCORE_MIN:
            continue

        in_rank, _, _ = calc_inout_for_stock(hist)
        if in_rank == "様子見":
            continue
        if close.iloc[-1] < ma20:
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score=mkt_score)
        rr = rr_info["rr"]
        if rr < SWING_RR_MIN:
            continue

        ev = expected_r_from_in_rank(in_rank, rr)
        if ev < SWING_EV_R_MIN:
            continue

        price = _safe_float(close.iloc[-1])
        entry = rr_info["entry"]
        gap = (price / entry - 1) * 100 if price > 0 else np.nan

        out.append(dict(
            ticker=ticker,
            name=str(r.get("name", ticker)),
            sector=str(r.get("sector", "")),
            score=score,
            rr=rr,
            ev=ev,
            in_rank=in_rank,
            entry=entry,
            price=price,
            gap=gap,
            tp_pct=rr_info["tp_pct"],
            sl_pct=rr_info["sl_pct"],
            tp_price=rr_info["tp_price"],
            sl_price=rr_info["sl_price"],
        ))

    out.sort(key=lambda x: (x["ev"], x["rr"], x["score"]), reverse=True)
    return out[:SWING_MAX_FINAL]


# ============================================================
# レポート
# ============================================================
def _fmt_pct(p: float) -> str:
    return f"{p*100:+.1f}%"


def build_report(today_str, today_date, mkt, pos_text, total_asset) -> str:
    mkt_score = int(mkt["score"])
    lev = 1.3 if mkt_score >= 50 else 1.1
    max_pos = int(total_asset * lev)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt['comment']})")
    lines.append(f"- レバ: {lev:.1f}倍")
    lines.append(f"- MAX建玉: 約{max_pos:,}円\n")

    lines.append("📈 セクター（5日）")
    for i, (s, p) in enumerate(top_sectors_5d(SECTOR_TOP_N)):
        lines.append(f"{i+1}. {s} ({p:+.2f}%)")
    lines.append("")

    lines.append("⚠ イベント")
    for e in build_event_warnings(today_date):
        lines.append(e)
    lines.append("")

    swing = run_swing(today_date, mkt_score)
    lines.append("🏆 Swing（順張り専用）")
    if swing:
        for c in swing:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R EV:{c['ev']:.2f}R IN:{c['in_rank']}")
            lines.append(f"  押し目IN:{c['entry']:.1f} / 現在:{c['price']:.1f} ({c['gap']:+.2f}%)")
            lines.append(f"  TP:{_fmt_pct(c['tp_pct'])} ({c['tp_price']:.1f})  SL:{_fmt_pct(c['sl_pct'])} ({c['sl_price']:.1f})\n")
    else:
        lines.append("- 該当なし\n")

    lines.append("📊 ポジション")
    lines.append(pos_text or "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信（★絶対に壊さない）
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print(text)
        return

    chunk = 3800
    for i in range(0, len(text), chunk):
        r = requests.post(WORKER_URL, json={"text": text[i:i+chunk]}, timeout=20)
        print("[LINE]", r.status_code)


# ============================================================
# Main
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