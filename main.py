from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.rr import compute_rr
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

RR_MIN = 1.8
SCORE_MIN_BASE = 60.0


# ============================================================
# events.csv 読み込み
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
    warns: List[str] = []
    events = load_events()
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
# Universe 読み込み & 決算除外
# ============================================================
def load_universe(path: str, today: date) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] universe file not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read universe: {e}")
        return pd.DataFrame()

    if "earnings_date" in df.columns:
        try:
            parsed = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
            df["earnings_date_parsed"] = parsed

            def ok(d):
                if not isinstance(d, date):
                    return True
                return abs((d - today).days) > EARNINGS_EXCLUDE_DAYS

            mask = parsed.apply(ok)
            df = df[mask]
        except Exception as e:
            print(f"[WARN] earnings filter failed: {e}")

    return df


# ============================================================
# レバレッジ
# ============================================================
def recommend_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 1.6, "強気（押し目＋一部ブレイク）"
    if mkt_score >= 60:
        return 1.4, "やや強め（押し目メイン）"
    if mkt_score >= 50:
        return 1.3, "中立（押し目）"
    if mkt_score >= 40:
        return 1.1, "やや守り"
    return 1.0, "守り（新規ごく少量）"


def calc_max_position(total_asset: float, lev: float) -> int:
    if not (np.isfinite(total_asset) and total_asset > 0 and lev > 0):
        return 0
    return int(round(total_asset * lev))


# ============================================================
# スクリーニング（Score × RR × ExpR）
# ============================================================
def run_screening(today: date, mkt_score: int) -> List[Dict]:
    uni = load_universe(UNIVERSE_PATH, today)
    if uni is None or uni.empty:
        return []

    # 地合いが悪いほどスコア基準を少し上げる
    dyn_score_min = SCORE_MIN_BASE + max(0.0, (50 - float(mkt_score)) * 0.2)
    dyn_score_min = float(np.clip(dyn_score_min, 55.0, 80.0))

    results: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get("ticker") or row.get("code") or "").strip()
        if not ticker:
            continue

        name = str(row.get("name", ticker))
        sector = str(row.get("sector", row.get("industry_big", "")))

        try:
            hist = yf.download(
                ticker,
                period="90d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            print(f"[WARN] download failed {ticker}: {e}")
            continue

        if hist is None or len(hist) < 60:
            continue

        # Quality スコア
        try:
            score = float(score_stock(ticker, hist, row))
        except Exception as e:
            print(f"[WARN] score_stock failed {ticker}: {e}")
            continue

        if not np.isfinite(score) or score < dyn_score_min:
            continue

        # RR 計算
        rr_info = compute_rr(hist, mkt_score)
        rr = float(rr_info.get("rr", 0.0))
        if not np.isfinite(rr) or rr < RR_MIN:
            continue

        entry = float(rr_info.get("entry", 0.0))
        tp_pct = float(rr_info.get("tp_pct", 0.0))
        sl_pct = float(rr_info.get("sl_pct", 0.0))

        exp_r = rr * (score / 100.0)  # 期待値Rの近似

        results.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "score": score,
                "rr": rr,
                "exp_r": exp_r,
                "entry": entry,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
            }
        )

    results.sort(key=lambda x: (x["exp_r"], x["rr"], x["score"]), reverse=True)
    return results[:MAX_FINAL_STOCKS]


# ============================================================
# レポート作成
# ============================================================
def build_report(
    today_str: str,
    today_date: date,
    mkt: Dict,
    pos_text: str,
    total_asset: float,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", ""))
    lev, lev_comment = recommend_leverage(mkt_score)

    if not (np.isfinite(total_asset) and total_asset > 0):
        total_asset = 2_000_000.0
    max_pos = calc_max_position(total_asset, lev)

    sectors = top_sectors_5d()
    events = build_event_warnings(today_date)
    core = run_screening(today_date, mkt_score)

    lines: List[str] = []

    # ヘッダー
    lines.append(f"📅 {today_str} stockbotTOM 日報")
    lines.append("")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- レバ: {lev:.1f}倍（{lev_comment}）")
    lines.append(f"- MAX建玉: 約{max_pos:,}円")
    lines.append("")

    # セクター
    if sectors:
        lines.append("📈 セクター（5日）")
        for i, (sec_name, chg) in enumerate(sectors[:5]):
            lines.append(f"{i+1}. {sec_name} ({chg:+.2f}%)")
        lines.append("")

    # イベント
    lines.append("⚠ イベント")
    if events:
        for ev in events:
            lines.append(ev)
    else:
        lines.append("- 特筆すべきイベントなし")
    lines.append("")

    # Core候補
    lines.append(f"🏆 Core候補（最大{MAX_FINAL_STOCKS}銘柄）")
    if core:
        for c in core:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(
                f"Score:{c['score']:.1f} RR:{c['rr']:.2f}R Exp:{c['exp_r']:.2f}R"
            )
            lines.append(
                f"IN:{c['entry']:.1f} TP:{c['tp_pct']*100:+.1f}% SL:{c['sl_pct']*100:.1f}%"
            )
            lines.append("")
    else:
        lines.append("- 該当なし")
        lines.append("")

    # ポジション
    lines.append("📊 ポジション")
    lines.append(pos_text.strip() or "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定（printのみ）")
        print(text)
        return

    chunk_size = 3900
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for ch in chunks:
        try:
            r = requests.post(
                WORKER_URL,
                json={"text": ch, "message": ch},
                timeout=15,
            )
            print("[LINE RESULT]", r.status_code, r.text)
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)


# ============================================================
# Main
# ============================================================
def main() -> None:
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = enhance_market_score()
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df)

    report = build_report(today_str, today_date, mkt, pos_text, total_asset)
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()