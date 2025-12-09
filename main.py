from __future__ import annotations
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import List, Dict

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.rr import compute_rr

UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
EARNINGS_EXCLUDE_DAYS = 3
MAX_FINAL_STOCKS = 3
WORKER_URL = os.getenv("WORKER_URL")


# ============================================================
# JST
# ============================================================
def jst_now() -> datetime:
    return datetime.now().astimezone(timezone(timedelta(hours=9)))


def jst_str() -> str:
    return jst_now().strftime("%Y-%m-%d")


# ============================================================
# 決算フィルタ
# ============================================================
def filter_earnings(df: pd.DataFrame, today: datetime) -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df

    try:
        df = df.copy()
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    except Exception:
        return df

    # 今日を Timestamp にして同じ型同士で引き算
    today_ts = pd.Timestamp(today.date())
    delta = (df["earnings_date"] - today_ts).dt.days.abs()

    return df[delta > EARNINGS_EXCLUDE_DAYS]


# ============================================================
# スクリーニング
# ============================================================
def run_screening(today: datetime, mkt_score: int) -> List[Dict]:
    df = pd.read_csv(UNIVERSE_PATH)
    df = filter_earnings(df, today)

    results = []
    for _, row in df.iterrows():
        ticker = str(row.get("code") or row.get("ticker") or "").strip()
        if not ticker:
            continue

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

            score = float(row.get("score", 0))
            rr_info = compute_rr(hist, mkt_score)

            results.append(
                dict(
                    ticker=ticker,
                    sector=row.get("sector", ""),
                    score=score,
                    rr=rr_info["rr"],
                    entry=rr_info["entry"],
                    tp_pct=rr_info["tp_pct"],
                    sl_pct=rr_info["sl_pct"],
                )
            )
        except Exception:
            continue

    # Score → RR の順でソート
    results.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)

    # RRフィルタ
    results = [r for r in results if r["rr"] >= 1.5]

    return results[:MAX_FINAL_STOCKS]


# ============================================================
# レポート作成
# ============================================================
def build_report(today: datetime, mkt: dict, pos_text: str, total_asset: float) -> str:
    mkt_score = int(mkt["score"])
    core = run_screening(today, mkt_score)
    sect = top_sectors_5d()
    lever = round(1.0 + (mkt_score - 50) / 100, 1)

    lines: list[str] = []
    lines.append(f"📅 {jst_str()} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({mkt.get('comment', '')})")
    lines.append(f"- レバ: {lever:.1f}倍（中立（押し目））")
    lines.append(f"- MAX建玉: 約{int(total_asset * lever):,}円\n")

    lines.append("📈 セクター（5日）")
    for i, (name, chg) in enumerate(sect[:5]):
        lines.append(f"{i+1}. {name} ({chg:+.2f}%)")
    lines.append("")

    lines.append("⚠ イベント")
    # events.csv を後でここに差し込む余地を残しておく
    lines.append("- 特になし\n")

    lines.append("🏆 Core候補（最大3銘柄）")
    if not core:
        lines.append("- 該当なし\n")
    else:
        for r in core:
            lines.append(f"- {r['ticker']} [{r['sector']}]")
            lines.append(f"Score:{r['score']:.1f} RR:{r['rr']:.2f}R")
            lines.append(
                f"IN:{r['entry']:.1f} "
                f"TP:+{r['tp_pct']*100:.1f}% "
                f"SL:{r['sl_pct']*100:.1f}%\n"
            )

    lines.append("📊 ポジション")
    lines.append(pos_text)

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return

    import requests

    try:
        # 長文対策で分割
        chunks = [text[i : i + 3800] for i in range(0, len(text), 3800)]
        for c in chunks:
            requests.post(WORKER_URL, json={"text": c}, timeout=10)
    except Exception:
        print(text)


# ============================================================
# Main
# ============================================================
def main():
    today = jst_now()
    mkt = calc_market_score()          # dict {"score": int, "comment": str}
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset = analyze_positions(pos_df)

    report = build_report(today, mkt, pos_text, total_asset)
    send_line(report)


if __name__ == "__main__":
    main()