from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.util import jst_today_str, jst_today_date
from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.rr import compute_rr

# ============================================================
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"      # あれば読む（無ければ無視）
WORKER_URL = os.getenv("WORKER_URL")

MAX_FINAL_STOCKS = 3
EARNINGS_EXCLUDE_DAYS = 3       # 決算 ±3日除外
LIQ_MIN_TURNOVER = 100_000_000  # 最低売買代金（20日平均）


# ============================================================
# イベント関連
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
        if not date_str or not label:
            continue
        events.append({"date": date_str, "label": label, "kind": kind})
    return events


def build_event_warnings(today: date) -> List[str]:
    events = load_events()
    warns: List[str] = []

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
            warns.append(f"⚠ {ev['label']}（{when}）")
    return warns


# ============================================================
# Universe / 決算フィルタ
# ============================================================
def load_universe(path: str = UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"[WARN] universe file not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read universe: {e}")
        return None

    if "ticker" not in df.columns:
        print("[WARN] universe has no 'ticker' column")
        return None

    df["ticker"] = df["ticker"].astype(str)

    # earnings_date を一度だけパース
    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row: pd.Series, today: date) -> bool:
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        delta = abs((d - today).days)
    except Exception:
        return False
    return delta <= EARNINGS_EXCLUDE_DAYS


# ============================================================
# スコア / RR の下限（地合い連動）
# ============================================================
def min_quality_threshold(mkt_score: int) -> float:
    if mkt_score >= 70:
        return 70.0
    if mkt_score >= 60:
        return 72.0
    if mkt_score >= 50:
        return 75.0
    if mkt_score >= 40:
        return 80.0
    return 82.0


def min_rr_threshold(mkt_score: int) -> float:
    if mkt_score >= 70:
        return 1.8
    if mkt_score >= 60:
        return 2.0
    if mkt_score >= 50:
        return 2.2
    if mkt_score >= 40:
        return 2.5
    return 2.8


# ============================================================
# レバレッジ推奨（地合い連動）
# ============================================================
def recommend_leverage(mkt_score: int) -> tuple[float, str]:
    if mkt_score >= 70:
        return 1.8, "強め（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return 1.6, "やや強め（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "中立（押し目メイン）"
    if mkt_score >= 40:
        return 1.1, "やや守り（ロット控えめ）"
    return 1.0, "守り（新規かなり絞る）"


# ============================================================
# yfinance ラッパ
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"[WARN] fetch history failed {ticker} (try {attempt+1}): {e}")
        time.sleep(0.8)
    return None


# ============================================================
# スクリーニング本体
# ============================================================
def run_screening(today: date, mkt_score: int) -> List[Dict]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

    q_min = min_quality_threshold(mkt_score)
    rr_min = min_rr_threshold(mkt_score)

    results: List[Dict] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        # 決算前後 ±N日 は除外
        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        # 流動性フィルタ
        if "Close" not in hist.columns or "Volume" not in hist.columns:
            continue
        close = hist["Close"].astype(float)
        vol = hist["Volume"].astype(float)
        if len(close) < 20:
            continue
        turnover = close * vol
        avg_turnover = float(turnover.rolling(20).mean().iloc[-1])
        if not np.isfinite(avg_turnover) or avg_turnover < LIQ_MIN_TURNOVER:
            continue

        # Quality スコア
        base_score = score_stock(ticker, hist, row)
        if base_score is None or not np.isfinite(base_score):
            continue
        if base_score < q_min:
            continue

        # RR / IN
        rr_info = compute_rr(hist, mkt_score)
        rr = float(rr_info.get("rr", 0.0))
        if not np.isfinite(rr) or rr < rr_min:
            continue

        entry = float(rr_info.get("entry", close.iloc[-1]))
        tp_pct = float(rr_info.get("tp_pct", 0.0))
        sl_pct = float(rr_info.get("sl_pct", 0.0))

        results.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "score": float(base_score),
                "rr": rr,
                "entry": entry,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
            }
        )

    results.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
    return results[:MAX_FINAL_STOCKS]


# ============================================================
# レポート構築
# ============================================================
def build_report(
    today_str: str,
    today_date: date,
    mkt: Dict,
    pos_text: str,
    total_asset: float,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    lev, lev_comment = recommend_leverage(mkt_score)
    if not np.isfinite(total_asset) or total_asset <= 0:
        total_asset = 3_000_000.0
    max_pos = int(round(total_asset * lev))

    # セクター
    secs = top_sectors_5d()
    sec_lines: List[str] = []
    for i, (name, chg) in enumerate(secs[:5]):
        sec_lines.append(f"{i+1}. {name} ({chg:+.2f}%)")

    # イベント
    event_lines = build_event_warnings(today_date)

    # スクリーニング
    core_list = run_screening(today_date, mkt_score)

    lines: List[str] = []

    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    lines.append("📈 セクター（5日）")
    if sec_lines:
        lines.extend(sec_lines)
    else:
        lines.append("- データ不足")
    lines.append("")

    lines.append("⚠ イベント")
    if event_lines:
        for ev in event_lines:
            lines.append(ev)
    else:
        lines.append("- 特になし")
    lines.append("")

    lines.append(f"🏆 Core候補（最大{MAX_FINAL_STOCKS}銘柄）")
    if core_list:
        for c in core_list:
            lines.append(
                f"- {c['ticker']} {c['name']} [{c['sector']}]"
            )
            lines.append(
                f"Score:{c['score']:.1f} RR:{c['rr']:.2f}R"
            )
            lines.append(
                f"IN:{c['entry']:.1f} TP:{c['tp_pct']*100:+.1f}% SL:{c['sl_pct']*100:.1f}%"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip() if pos_text.strip() else "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信（分割対応）
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定。print のみ。")
        print(text)
        return

    chunk_size = 3900
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=15)
            print("[LINE RESULT]", r.status_code, r.text[:200])
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)
            print(ch)


# ============================================================
# main
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    # 地合い（半導体込み強化版）
    mkt = enhance_market_score()

    # ポジション
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df)

    # レポート作成
    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        pos_text=pos_text,
        total_asset=total_asset,
    )

    # ログ & LINE
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()