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
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

SCREENING_TOP_N = 10
MAX_FINAL_STOCKS = 3

EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# 共通ユーティリティ
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    events = []
    for _, r in df.iterrows():
        d = str(r.get("date", "")).strip()
        label = str(r.get("label", "")).strip()
        kind = str(r.get("kind", "")).strip()
        if d and label:
            events.append({"date": d, "label": label, "kind": kind})
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
            warns.append(f"⚠ {ev['label']}（{when}）: ロット注意")
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
        return delta <= EARNINGS_EXCLUDE_DAYS
    except Exception:
        return False


# ============================================================
# Data fetch
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(1.0)
    return None


# ============================================================
# ダイナミック最低スコア
# ============================================================
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


# ============================================================
# レバ計算
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 1.8, "強め"
    if mkt_score >= 60:
        return 1.5, "やや強め"
    if mkt_score >= 50:
        return 1.3, "標準"
    if mkt_score >= 40:
        return 1.1, "やや守り"
    return 1.0, "守り"


def calc_max_position(total_asset: float, lev: float) -> int:
    return int(round(total_asset * lev))


# ============================================================
# 市場スコア拡張(SOX/NVDA)
# ============================================================
def enhance_market_score() -> Dict:
    mkt = calc_market_score()
    score = float(mkt.get("score", 50))

    try:
        sox = yf.Ticker("^SOX").history(period="5d")
        if sox is not None and not sox.empty:
            sox_chg = float(sox["Close"].iloc[-1] / sox["Close"].iloc[0] - 1.0) * 100
            score += np.clip(sox_chg / 2.0, -5, 5)
    except Exception:
        pass

    try:
        nvda = yf.Ticker("NVDA").history(period="5d")
        if nvda is not None and not nvda.empty:
            nvda_chg = float(nvda["Close"].iloc[-1] / nvda["Close"].iloc[0] - 1.0) * 100
            score += np.clip(nvda_chg / 3.0, -4, 4)
    except Exception:
        pass

    score = float(np.clip(round(score), 0, 100))
    mkt["score"] = int(score)
    return mkt


# ============================================================
# Screening
# ============================================================
def run_screening(today: datetime.date, mkt_score: int) -> List[Dict]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

    min_score = dynamic_min_score(mkt_score)
    raw_list = []

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

        if base_score < min_score:
            continue

        tp_pct, sl_pct, rr = compute_tp_sl_rr(hist, mkt_score)
        if rr < 1.50:
            continue

        price = float(hist["Close"].iloc[-1])
        entry = price * 0.995
        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)

        raw_list.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "price": price,
                "score": base_score,
                "entry": entry,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "rr": rr,
            }
        )

    raw_list.sort(key=lambda x: (x["rr"], x["score"]), reverse=True)
    return raw_list[:MAX_FINAL_STOCKS]


# ============================================================
# レポート
# ============================================================
def build_report(today_str: str, today_date: datetime.date, mkt: Dict, total_asset: float, pos_text: str) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))

    lev, lev_comment = recommend_leverage(mkt_score)
    est_asset = total_asset if total_asset > 0 else 2_000_000
    max_pos = calc_max_position(est_asset, lev)

    secs = top_sectors_5d()
    if secs:
        sec_text = "\n".join([f"{i+1}. {n} ({c:+.2f}%)" for i, (n, c) in enumerate(secs)])
    else:
        sec_text = "データ不足"

    event_lines = build_event_warnings(today_date)
    if not event_lines:
        event_lines = ["- 特筆すべきイベントなし"]

    core_list = run_screening(today_date, mkt_score)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円\n")

    lines.append("📈 セクター（5日）")
    lines.append(sec_text + "\n")

    lines.append("⚠ イベント")
    lines.extend(event_lines)
    lines.append("")

    lines.append(f"🏆 Core候補（最大{MAX_FINAL_STOCKS}銘柄）")
    if not core_list:
        lines.append("- レベル相当の候補なし\n")
    else:
        for c in core_list:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"Score:{c['score']:.1f} RR:{c['rr']:.2f}R")
            lines.append(f"IN:{c['entry']:.1f} TP:+{c['tp_pct']*100:.1f}% SL:{c['sl_pct']*100:.1f}%\n")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip())

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL未設定")
        print(text)
        return

    chunk = 3800
    for i in range(0, len(text), chunk):
        ch = text[i:i+chunk]
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=10)
            print("[LINE]", r.status_code)
        except Exception as e:
            print("[ERROR LINE]", e)


# ============================================================
# Entry
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, *_ = analyze_positions(pos_df)
    if total_asset <= 0 or not np.isfinite(total_asset):
        total_asset = 2_000_000

    report = build_report(today_str, today_date, mkt, total_asset, pos_text)
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()