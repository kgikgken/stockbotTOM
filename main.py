from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Tuple, Optional

import pandas as pd
import yfinance as yf
import numpy as np
import requests

from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.rr import compute_rr
from utils.util import jst_today_str, jst_today_date

# ============================================================
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

# スクリーニング設定
SCREENING_TOP_N = 30          # まずスコア上位30銘柄まで見る
MAX_FINAL_STOCKS = 3          # LINEに出す最大銘柄数
EARNINGS_EXCLUDE_DAYS = 3     # 決算±3日除外
MIN_RR = 1.8                  # 最低でもこのRR未満は捨てる
MIN_SCORE = 60.0              # スコアの最低ライン


# ============================================================
# 日付 / イベント関連
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
        if not date_str or not label:
            continue
        kind = str(row.get("kind", "")).strip()
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
# 決算除外
# ============================================================
def in_earnings_window(row: pd.Series, today: date) -> bool:
    d_raw = row.get("earnings_date_parsed", None)
    if d_raw is None or (isinstance(d_raw, float) and np.isnan(d_raw)):
        return False
    try:
        return abs((d_raw - today).days) <= EARNINGS_EXCLUDE_DAYS
    except Exception:
        return False


# ============================================================
# レバレッジ
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 1.8, "攻め（強め）"
    if mkt_score >= 60:
        return 1.5, "やや攻め（押し目＋一部ブレイク）"
    if mkt_score >= 50:
        return 1.3, "中立（押し目メイン）"
    if mkt_score >= 40:
        return 1.1, "やや守り（ロット控えめ）"
    return 1.0, "守り（新規小ロット）"


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# Universe 読み込み
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

    # earnings_dateを一度だけパースしておく
    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


# ============================================================
# スクリーニング本体
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    for attempt in range(2):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df is not None and len(df) >= 60:
                return df
        except Exception as e:
            print(f"[WARN] fetch_history failed {ticker} try{attempt+1}: {e}")
    return None


def run_screening(today: date, mkt_score: int) -> List[Dict]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

    candidates: List[Dict] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        # 決算ウィンドウ除外
        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None:
            continue

        base_score = score_stock(ticker, hist, row)
        if not np.isfinite(base_score) or base_score < MIN_SCORE:
            continue

        rr_info = compute_rr(hist, mkt_score)
        rr = float(rr_info.get("rr", 0.0))
        if not np.isfinite(rr) or rr < MIN_RR:
            continue

        candidates.append(
            dict(
                ticker=ticker,
                name=name,
                sector=sector,
                score=float(base_score),
                rr=rr,
                entry=float(rr_info["entry"]),
                tp_pct=float(rr_info["tp_pct"]),
                sl_pct=float(rr_info["sl_pct"]),
            )
        )

    # スコア→RRの優先度でソート
    candidates.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)

    return candidates[:MAX_FINAL_STOCKS]


# ============================================================
# レポート作成
# ============================================================
def build_report() -> str:
    today_str = jst_today_str()
    today_date = jst_today_date()

    # 地合い
    mkt_info = enhance_market_score()
    mkt_score = int(mkt_info.get("score", 50))
    mkt_comment = str(mkt_info.get("comment", "中立"))

    # ポジション / レバ&建玉
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df)

    lev, lev_comment = recommend_leverage(mkt_score)
    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 3_000_000.0
    max_pos = calc_max_position(total_asset, lev)

    # セクター上位
    sectors = top_sectors_5d()
    # イベント
    events = build_event_warnings(today_date)

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
    if sectors:
        for i, (sec_name, chg) in enumerate(sectors[:5]):
            lines.append(f"{i+1}. {sec_name} ({chg:+.2f}%)")
    else:
        lines.append("- データなし")
    lines.append("")

    lines.append("⚠ イベント")
    if events:
        for ev in events:
            lines.append(ev)
    else:
        lines.append("- 特になし")
    lines.append("")

    lines.append(f"🏆 Core候補（最大{MAX_FINAL_STOCKS}銘柄）")
    if not core_list:
        lines.append("- 該当なし")
    else:
        for c in core_list:
            lines.append(
                f"- {c['ticker']} {c['name']} [{c['sector']}]"
            )
            lines.append(
                f"Score:{c['score']:.1f} RR:{c['rr']:.2f}R"
            )
            lines.append(
                f"IN:{c['entry']:.1f} "
                f"TP:{c['tp_pct']*100:+.1f}% "
                f"SL:{c['sl_pct']*100:.1f}%"
            )
            lines.append("")
    lines.append("")
    lines.append("📊 ポジション")
    lines.append(pos_text.strip() or "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定。以下の内容をprintのみ。")
        print(text)
        return

    chunk_size = 3900
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=15)
            print("[LINE]", r.status_code, r.text[:200])
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)
            print(ch)


# ============================================================
# Entry Point
# ============================================================
def main() -> None:
    report = build_report()
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()