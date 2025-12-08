from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.rr import compute_tp_sl_rr
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
EARNINGS_EXCLUDE_DAYS = 3

SWAP_RR_DIFF_THRESHOLD = 0.8  # 新規候補とのRR差がこれ以上なら乗り換え候補


# ============================================================
# 日付関連
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ============================================================
# Event
# ============================================================
def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    events: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        date_str = str(row.get("date", "")).strip()
        label = str(row.get("label", "")).strip()
        kind = str(row.get("kind", "")).strip()
        if date_str and label:
            events.append({"date": date_str, "label": label, "kind": kind})
    return events


def build_event_warnings(today: datetime.date) -> List[str]:
    events = load_events()
    out: List[str] = []
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
            out.append(f"⚠ {ev['label']}（{when}）")
    return out


# ============================================================
# Universe
# ============================================================
def load_universe() -> Optional[pd.DataFrame]:
    if not os.path.exists(UNIVERSE_PATH):
        return None
    try:
        df = pd.read_csv(UNIVERSE_PATH)
    except Exception:
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
        delta = abs((d - today).days)
    except Exception:
        return False
    return delta <= EARNINGS_EXCLUDE_DAYS


# ============================================================
# Data fetch
# ============================================================
def fetch_history(ticker: str) -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period="130d")
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(1.0)
    return None


# ============================================================
# Screening
# ============================================================
def run_screening(today: datetime.date, mkt_score: int) -> List[Dict]:
    df = load_universe()
    if df is None:
        return []

    # 地合いで最低ライン変化
    if mkt_score >= 70:
        min_score = 72.0
    elif mkt_score >= 60:
        min_score = 75.0
    elif mkt_score >= 50:
        min_score = 78.0
    elif mkt_score >= 40:
        min_score = 80.0
    else:
        min_score = 82.0

    cands: List[Dict] = []

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
        if base_score is None or base_score < min_score:
            continue

        rr_info = compute_tp_sl_rr(hist, mkt_score)
        rr_value = float(rr_info.get("rr", 0.0))  # 数値（比較用）

        # RRフィルタ
        if rr_value < 1.5:
            continue

        # 表示用文字列
        rr_str = f"{rr_value:.2f}R"

        # entry/tp/sl
        entry = rr_info.get("entry", 0.0)
        tp_pct = rr_info.get("tp_pct", 0.08)
        sl_pct = rr_info.get("sl_pct", -0.04)

        # 今日 vs 数日後
        price = float(hist["Close"].iloc[-1])
        gap = abs(price - entry) / price if price > 0 else 1.0
        entry_type = "today" if gap <= 0.01 else "soon"

        cands.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": float(base_score),
            "rr_value": rr_value,
            "rr": rr_str,
            "price": price,
            "entry": entry,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "entry_type": entry_type
        })

    # RR優先、同一RRならスコア順
    cands.sort(key=lambda x: (x["rr_value"], x["score"]), reverse=True)
    return cands[:MAX_FINAL_STOCKS]


# ============================================================
# RRスワップ候補
# ============================================================
def find_swap_candidates(
    pos_df: Optional[pd.DataFrame],
    mkt_score: int,
    core_list: List[Dict]
) -> List[Dict]:
    if pos_df is None or pos_df.empty:
        return []

    if not core_list:
        return []

    results: List[Dict] = []

    for _, row in pos_df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        rr_pos_info = compute_tp_sl_rr(hist, mkt_score)
        rr_pos = float(rr_pos_info.get("rr", 0.0))

        best_new = None
        best_diff = 0.0

        for c in core_list:
            rr_new = float(c.get("rr_value", 0.0))
            diff = rr_new - rr_pos
            if diff > best_diff:
                best_diff = diff
                best_new = c

        if best_new is not None and best_diff >= SWAP_RR_DIFF_THRESHOLD:
            results.append({
                "from_ticker": ticker,
                "from_rr": rr_pos,
                "to_ticker": best_new["ticker"],
                "to_name": best_new["name"],
                "to_rr": float(best_new["rr_value"]),
                "diff": best_diff
            })

    return results


# ============================================================
# Report
# ============================================================
def build_report(
    today_str: str,
    today: datetime.date,
    mkt: Dict,
    total_asset: float,
    pos_text: str,
    pos_df: Optional[pd.DataFrame],
) -> str:
    ms = int(mkt.get("score", 50))
    comment = str(mkt.get("comment", ""))

    if ms >= 70:
        lev = 1.8
        note = "強い（押し目＋ブレイク）"
    elif ms >= 60:
        lev = 1.5
        note = "やや強く"
    elif ms >= 50:
        lev = 1.3
        note = "中立（押し目）"
    elif ms >= 40:
        lev = 1.1
        note = "弱め"
    else:
        lev = 1.0
        note = "守り"

    max_pos = int(total_asset * lev)

    secs = top_sectors_5d()
    sec_lines = [f"{i+1}. {s} ({c:+.2f}%)" for i, (s, c) in enumerate(secs[:5])]
    sec_text = "\n".join(sec_lines)

    ev = build_event_warnings(today)
    if not ev:
        ev = ["- 特になし"]

    core = run_screening(today, ms)
    swap_list = find_swap_candidates(pos_df, ms, core)

    out: List[str] = []
    out.append(f"📅 {today_str} stockbotTOM 日報\n")

    # 結論
    out.append("◆ 今日の結論")
    out.append(f"- 地合い: {ms}点 ({comment})")
    out.append(f"- レバ: {lev:.1f}倍（{note}）")
    out.append(f"- MAX建玉: 約{max_pos:,}円\n")

    # セクター簡略
    out.append("📈 セクター（5日）")
    out.append(sec_text + "\n")

    # イベント
    out.append("⚠ イベント")
    for e in ev:
        out.append(e)
    out.append("")

    # Core候補
    out.append(f"🏆 Core候補（最大{MAX_FINAL_STOCKS}銘柄）")
    if not core:
        out.append("なし\n")
    else:
        for c in core:
            out.append(f"- {c['ticker']}.T {c['name']} [{c['sector']}]")
            out.append(f"Score:{c['score']:.1f} RR:{c['rr']}")
            out.append(
                f"IN:{c['entry']:.1f} TP:{c['tp_pct']*100:.1f}% SL:{c['sl_pct']*100:.1f}%"
            )
            out.append("")

    # スワップ候補
    out.append("🔄 ポジション入れ替え候補（RR差ベース）")
    if not swap_list:
        out.append("- 明確な入れ替え候補なし\n")
    else:
        for s in swap_list:
            out.append(
                f"- {s['from_ticker']} (現RR:{s['from_rr']:.2f}R) → "
                f"{s['to_ticker']} {s['to_name']} (新RR:{s['to_rr']:.2f}R, 差:+{s['diff']:.2f}R)"
            )
        out.append("")

    # ポジション
    out.append("📊 ポジション")
    out.append(pos_text.strip())
    return "\n".join(out)


# ============================================================
# LINE
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print(text)
        return

    chunks = [text[i:i+3800] for i in range(0, len(text), 3800)] or [text]
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

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev_pos, risk = analyze_positions(pos_df)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000

    report = build_report(today_str, today, mkt, total_asset, pos_text, pos_df)
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()