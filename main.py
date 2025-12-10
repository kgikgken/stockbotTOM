from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions_with_rr
from utils.scoring import score_stock, compute_in_rank
from utils.rr import compute_rr, rr_min_by_market
from utils.util import jst_today_str, jst_today_date


# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"
WORKER_URL = os.getenv("WORKER_URL")

EARNINGS_EXCLUDE_DAYS = 3
MAX_FINAL_STOCKS = 3


# ============================================================
# 日付 / イベント
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
        d = str(row.get("date", "")).strip()
        label = str(row.get("label", "")).strip()
        kind = str(row.get("kind", "")).strip()
        if not d or not label:
            continue
        events.append({"date": d, "label": label, "kind": kind})
    return events


def build_event_warnings(today: datetime.date) -> List[str]:
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
# Universe ロード & 決算フィルタ
# ============================================================
def load_universe(path: str = UNIVERSE_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"[WARN] universe not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read universe: {e}")
        return None

    # ticker カラム必須
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
# 地合い関連
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 2.0, "強気（ブレイク＋押し目）"
    if mkt_score >= 60:
        return 1.7, "やや強気（押し目＋一部ブレイク）"
    if mkt_score >= 50:
        return 1.3, "中立（押し目メイン）"
    if mkt_score >= 40:
        return 1.1, "やや守り（新規ロット小さめ）"
    return 1.0, "守り（新規は最小ロット）"


# ============================================================
# スクリーニング
# ============================================================
def fetch_history(ticker: str, period: str = "130d") -> Optional[pd.DataFrame]:
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"[WARN] history fail {ticker} (try {attempt+1}): {e}")
    return None


def classify_no_candidate_reason(stats: Dict[str, int]) -> str:
    # stats: {"total":..., "earnings":..., "score":..., "in":..., "rr":...}
    if stats["total"] == 0:
        return "ユニバースに銘柄がありません"
    if stats["earnings"] == stats["total"]:
        return "決算前後（±3日）が多く、安全のため除外"
    if stats["score"] == stats["total"]:
        return "形不足（トレンド未形成）"
    if stats["in"] == stats["total"]:
        return "INゾーン未達（押し目前）"
    if stats["rr"] == stats["total"]:
        return "RR不足（リスクリワードが足りない）"
    # 混在パターン
    if stats["in"] > 0:
        return "INゾーンに入るまで待つ日"
    if stats["rr"] > 0:
        return "RRが伸びきるまで待つ日"
    return "総合的に期待値が足りない日"


def run_screening(
    today: datetime.date,
    mkt_score: int,
) -> Tuple[List[Dict], Optional[str]]:
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return [], "ユニバースファイル未読込"

    rr_min = rr_min_by_market(mkt_score)

    stats = {"total": 0, "earnings": 0, "score": 0, "in": 0, "rr": 0}

    candidates: List[Dict] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue
        stats["total"] += 1

        # 決算前後は除外
        if in_earnings_window(row, today):
            stats["earnings"] += 1
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "不明")))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        # ① 形のスコア（0〜100）
        score = score_stock(hist)
        if not np.isfinite(score) or score < 60.0:
            stats["score"] += 1
            continue

        # ② INゾーン判定
        in_rank = compute_in_rank(hist)
        if in_rank == "様子見":
            stats["in"] += 1
            continue

        # ③ RR計算
        rr_info = compute_rr(hist, mkt_score, in_rank=in_rank)
        rr = float(rr_info["rr"])
        if not np.isfinite(rr) or rr < rr_min:
            stats["rr"] += 1
            continue

        candidates.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "score": float(score),
                "in_rank": in_rank,
                "rr": rr,
                "entry": float(rr_info["entry"]),
                "tp_pct": float(rr_info["tp_pct"]),
                "sl_pct": float(rr_info["sl_pct"]),
            }
        )

    candidates.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
    top = candidates[:MAX_FINAL_STOCKS]

    if not top:
        reason = classify_no_candidate_reason(stats)
    else:
        reason = None

    return top, reason


# ============================================================
# レポート作成
# ============================================================
def build_report(
    today_str: str,
    today_date: datetime.date,
    mkt: Dict,
    pos_text: str,
    total_asset: float,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    base_comment = str(mkt.get("comment", ""))
    # 地合いコメントを少し人間語に
    if mkt_score >= 70:
        detail = "買い優勢（ブレイク＋押し目）"
    elif mkt_score >= 60:
        detail = "押し目適正（一部ブレイク可）"
    elif mkt_score >= 50:
        detail = "中立（押し目メイン）"
    elif mkt_score >= 40:
        detail = "弱い押し目（新規小さく）"
    else:
        detail = "守り（新規ほぼ見送り）"

    lev, lev_comment = recommend_leverage(mkt_score)
    est_asset = total_asset if np.isfinite(total_asset) and total_asset > 0 else 2_000_000.0
    max_pos = int(round(est_asset * lev))

    # セクター
    secs = top_sectors_5d()
    sec_lines: List[str] = []
    for i, (name, chg) in enumerate(secs[:5]):
        sec_lines.append(f"{i+1}. {name} ({chg:+.2f}%)")

    # イベント
    event_lines = build_event_warnings(today_date)

    # スクリーニング
    core_list, no_cand_reason = run_screening(today_date, mkt_score)

    lines: List[str] = []

    # 結論
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({base_comment})")
    lines.append(f"  ※{detail}")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    # セクター
    lines.append("📈 セクター（5日）")
    if sec_lines:
        lines.extend(sec_lines)
    else:
        lines.append("データ不足")
    lines.append("")

    # イベント
    if event_lines:
        lines.append("⚠ イベント")
        for s in event_lines:
            lines.append(s)
        lines.append("")

    # Core候補
    lines.append(f"🏆 Core候補（最大{MAX_FINAL_STOCKS}銘柄）")
    if core_list:
        for c in core_list:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(
                f"  Score:{c['score']:.1f} RR:{c['rr']:.2f}R IN:{c['in_rank']}"
            )
            lines.append(
                f"  IN:{c['entry']:.1f} "
                f"TP:+{c['tp_pct']*100:.1f}% "
                f"SL:{c['sl_pct']*100:.1f}%"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        if no_cand_reason:
            lines.append("")
            lines.append("📌 該当なし理由")
            lines.append(f"- {no_cand_reason}")
        lines.append("")

    # ポジション
    lines.append("📊 ポジション")
    lines.append(pos_text.strip() or "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信（分割）
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定。printのみ。")
        print(text)
        return

    chunk_size = 3900
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=15)
            print("[LINE RESULT]", r.status_code, r.text)
        except Exception as e:
            print("[ERROR] LINE送信エラー:", e)
            print(ch)


# ============================================================
# main
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    # 地合い（強化版）
    mkt = enhance_market_score()

    # ポジション & 資産 & RR
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, _rr_map = analyze_positions_with_rr(pos_df, mkt_score=int(mkt.get("score", 50)))

    report = build_report(
        today_str=today_str,
        today_date=today_date,
        mkt=mkt,
        pos_text=pos_text,
        total_asset=total_asset,
    )

    print(report)
    send_line(report)


if __name__ == "__main__":
    main()