from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.rr import compute_rr
from utils.scoring import score_stock

# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

EARNINGS_EXCLUDE_DAYS = 3
MAX_FINAL_STOCKS = 3
SCORE_MIN = 65.0   # スコア最低ライン
RR_MIN = 2.0       # RR最低ライン


# ============================================================
# 日付系
# ============================================================
def jst_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


def jst_today_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


def jst_today_date() -> datetime.date:
    return jst_now().date()


# ============================================================
# イベント関連
# ============================================================
def load_events(path: str = EVENTS_PATH) -> List[Dict[str, str]]:
    """
    events.csv -> [{"date": "2025-12-13", "label": "FOMC", "kind": "macro"}, ...]
    無ければ空リスト。
    """
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


def build_event_warnings(today: datetime.date) -> List[str]:
    """
    イベントの2日前〜翌日までを警戒表示。
    """
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

    if not warns:
        warns.append("- 特になし")
    return warns


# ============================================================
# Earnings 除外
# ============================================================
def filter_earnings(df: pd.DataFrame, today: datetime.date) -> pd.DataFrame:
    """
    決算日 ±EARNINGS_EXCLUDE_DAYS 日は除外。
    earnings_date は date で扱う（tz混在エラー防止）。
    """
    if "earnings_date" not in df.columns:
        return df

    try:
        ed = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    except Exception:
        return df

    delta = (ed - today).abs()
    mask = delta > EARNINGS_EXCLUDE_DAYS
    df = df.copy()
    df["earnings_date_parsed"] = ed
    return df[mask]


# ============================================================
# スクリーニング
# ============================================================
def run_screening(today: datetime.date, mkt_score: int) -> List[Dict]:
    try:
        df = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    df = filter_earnings(df, today)

    ticker_col = "ticker" if "ticker" in df.columns else "code"
    if ticker_col not in df.columns:
        return []

    df[ticker_col] = df[ticker_col].astype(str).str.strip()

    candidates: List[Dict] = []

    for _, row in df.iterrows():
        ticker = row.get(ticker_col, "")
        if not ticker or ticker == "nan":
            continue

        try:
            hist = yf.download(
                ticker,
                period="130d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception:
            continue

        if hist is None or len(hist) < 60:
            continue

        # Coreスコア
        try:
            score = float(score_stock(hist))
        except Exception:
            continue

        if not np.isfinite(score) or score < SCORE_MIN:
            continue

        # RR
        rr_info = compute_rr(hist, mkt_score)
        rr = float(rr_info.get("rr", 0.0))
        if not np.isfinite(rr) or rr < RR_MIN:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "")))

        candidates.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=score,
                rr=rr,
                entry=float(rr_info.get("entry", 0.0)),
                tp_pct=float(rr_info.get("tp_pct", 0.0)),
                sl_pct=float(rr_info.get("sl_pct", 0.0)),
            )
        )

    # RR → Score の順でソート（両方大きいものを優先）
    candidates.sort(key=lambda x: (x["rr"], x["score"]), reverse=True)
    return candidates[:MAX_FINAL_STOCKS]


# ============================================================
# レポート作成
# ============================================================
def build_report(
    today_str: str,
    today: datetime.date,
    mkt: Dict,
    pos_text: str,
    total_asset: float,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    # レバ推奨
    if mkt_score >= 70:
        lev = 1.8
        lev_comment = "強め（押し目＋ブレイク）"
    elif mkt_score >= 60:
        lev = 1.5
        lev_comment = "やや強め（押し目メイン）"
    elif mkt_score >= 50:
        lev = 1.3
        lev_comment = "中立（押し目）"
    elif mkt_score >= 40:
        lev = 1.1
        lev_comment = "やや守り"
    else:
        lev = 1.0
        lev_comment = "守り"

    # 資産推定
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 3_000_000.0
    max_pos = int(round(total_asset * lev))

    # セクター
    sectors = top_sectors_5d()
    sector_lines: List[str] = []
    for i, (name, chg) in enumerate(sectors[:5]):
        sector_lines.append(f"{i+1}. {name} ({chg:+.2f}%)")
    if not sector_lines:
        sector_lines.append("データ不足")

    # イベント
    event_lines = build_event_warnings(today)

    # スクリーニング
    core_list = run_screening(today, mkt_score)

    lines: List[str] = []

    # ヘッダ
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    # セクター
    lines.append("📈 セクター（5日）")
    lines.extend(sector_lines)
    lines.append("")

    # イベント
    lines.append("⚠ イベント")
    lines.extend(event_lines)
    lines.append("")

    # Core候補
    lines.append("🏆 Core候補（最大3銘柄）")
    if core_list:
        for c in core_list:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"Score:{c['score']:.1f} RR:{c['rr']:.2f}R")
            lines.append(
                f"IN:{c['entry']:.1f} "
                f"TP:{c['tp_pct']*100:+.1f}% "
                f"SL:{c['sl_pct']*100:.1f}%"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    # ポジション
    lines.append("📊 ポジション")
    lines.append(pos_text or "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(msg: str) -> None:
    if not WORKER_URL:
        print(msg)
        return
    try:
        requests.post(WORKER_URL, json={"text": msg}, timeout=10)
    except Exception:
        print(msg)


# ============================================================
# Main
# ============================================================
def main() -> None:
    today_date = jst_today_date()
    today_str = jst_today_str()

    mkt = calc_market_score()
    if isinstance(mkt, dict):
        mkt_dict = mkt
    else:
        mkt_dict = {"score": int(mkt), "comment": "中立"}

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df)

    report = build_report(today_str, today_date, mkt_dict, pos_text, total_asset)
    send_line(report)


if __name__ == "__main__":
    main()