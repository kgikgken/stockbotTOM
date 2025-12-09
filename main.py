from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict

import pandas as pd
import yfinance as yf
import requests

from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.rr import compute_rr
from utils.util import jst_today_str, jst_today_date

# ============================================================
# 設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EVENTS_PATH = "events.csv"  # まだ未使用（将来用）
WORKER_URL = os.getenv("WORKER_URL")

EARNINGS_EXCLUDE_DAYS = 3
MAX_FINAL_STOCKS = 3


# ============================================================
# Earnings 除外
# ============================================================
def filter_earnings(df: pd.DataFrame, today_date) -> pd.DataFrame:
    """決算 ±N日 の銘柄を除外"""
    if "earnings_date" not in df.columns:
        return df

    try:
        ed = pd.to_datetime(df["earnings_date"], errors="coerce")
        today_ts = pd.Timestamp(today_date)
        delta = (ed - today_ts).dt.days.abs()
        mask = (delta.isna()) | (delta > EARNINGS_EXCLUDE_DAYS)
        return df[mask]
    except Exception:
        return df


# ============================================================
# スクリーニング
# ============================================================
from utils.scoring import score_stock  # 上で import すると循環もないのでここでOK


def run_screening(today_date, mkt_score: int) -> List[Dict]:
    """universe_jpx.csv → Core候補を返す"""
    if not os.path.exists(UNIVERSE_PATH):
        return []

    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return []

    # 列名: ticker 想定（1332.T みたいな形式）
    if "ticker" not in uni.columns:
        return []

    uni = filter_earnings(uni, today_date)

    results: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        # 株価履歴
        try:
            hist = yf.download(
                ticker,
                period="60d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if hist is None or len(hist) < 40:
                continue
        except Exception:
            continue

        # Coreスコア
        try:
            score = float(score_stock(ticker, hist, row))
        except Exception:
            score = 0.0

        # RR計算
        rr_info = compute_rr(hist, mkt_score)
        rr = float(rr_info.get("rr", 0.0))
        if rr < 1.5:
            continue

        results.append(
            dict(
                ticker=ticker,
                sector=str(row.get("sector", row.get("industry_big", ""))),
                score=score,
                rr=rr,
                entry=float(rr_info.get("entry", 0.0)),
                tp_pct=float(rr_info.get("tp_pct", 0.0)),
                sl_pct=float(rr_info.get("sl_pct", 0.0)),
            )
        )

    # スコア → RR の順でソート
    results.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
    return results[:MAX_FINAL_STOCKS]


# ============================================================
# レポート作成
# ============================================================
def build_report(
    today_str: str,
    today_date,
    mkt: Dict,
    pos_text: str,
    total_asset: float,
) -> str:
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    core = run_screening(today_date, mkt_score)
    sectors = top_sectors_5d()

    # レバとMAX建玉
    lev = 1.0 + (mkt_score - 50) / 100.0
    lev = max(1.0, round(lev, 1))
    max_pos = int(total_asset * lev)

    lines: List[str] = []

    lines.append(f"📅 {today_str} stockbotTOM 日報\n")

    # 結論
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt_comment})")
    lines.append(f"- レバ: {lev:.1f}倍（中立（押し目））")
    lines.append(f"- MAX建玉: 約{max_pos:,}円\n")

    # セクター
    lines.append("📈 セクター（5日）")
    for i, (name, chg) in enumerate(sectors[:5]):
        lines.append(f"{i+1}. {name} ({chg:+.2f}%)")
    lines.append("")

    # Core候補
    lines.append("🏆 Core候補（最大3銘柄）")
    if core:
        for r in core:
            lines.append(f"- {r['ticker']} [{r['sector']}]")
            lines.append(
                f"Score:{r['score']:.1f} RR:{r['rr']:.2f}R"
            )
            lines.append(
                f"IN:{r['entry']:.1f} TP:{r['tp_pct']*100:+.1f}% SL:{r['sl_pct']*100:.1f}%\n"
            )
    else:
        lines.append("- 該当なし\n")

    # ポジション
    lines.append("📊 ポジション")
    lines.append(pos_text if pos_text else "ノーポジション")

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(msg: str) -> None:
    if not WORKER_URL:
        print(msg)
        return
    try:
        # Worker は {"text": "..."} で受ける前提
        requests.post(WORKER_URL, json={"text": msg}, timeout=10)
    except Exception:
        print(msg)


# ============================================================
# Main
# ============================================================
def main() -> None:
    today_date = jst_today_date()
    today_str = jst_today_str()

    mkt = enhance_market_score()
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df)

    report = build_report(today_str, today_date, mkt, pos_text, total_asset)
    send_line(report)


if __name__ == "__main__":
    main()