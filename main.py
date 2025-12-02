from __future__ import annotations

import os
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


UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算日 ±N日を除外
EARNINGS_EXCLUDE_DAYS = 3


# ============================================================
# 日付ユーティリティ
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ============================================================
# Universe 読み込み
# ============================================================
def load_universe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except:
        return None

    if "ticker" not in df.columns:
        return None

    df["ticker"] = df["ticker"].astype(str)

    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row, today: datetime.date) -> bool:
    d = row.get("earnings_date_parsed")
    if pd.isna(d):
        return False
    try:
        return abs((d - today).days) <= EARNINGS_EXCLUDE_DAYS
    except:
        return False


# ============================================================
# 株価データ取得
# ============================================================
def fetch_history(ticker: str, period="130d"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# ============================================================
# スクリーニング
# ============================================================
def calc_tp_sl(price: float, vola: float, mkt_score: int):
    if vola < 0.02:
        tp, sl = 0.08, -0.03
    elif vola > 0.06:
        tp, sl = 0.12, -0.06
    else:
        tp, sl = 0.10, -0.04

    if mkt_score >= 70:
        tp += 0.02
    if mkt_score < 45:
        tp -= 0.02
        sl = max(sl, -0.03)

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))

    return tp, sl


def run_screening(today: datetime.date, mkt_score: int):
    df = load_universe(UNIVERSE_PATH)
    if df is None:
        return []

    results = []

    for _, row in df.iterrows():
        ticker = row["ticker"]

        if in_earnings_window(row, today):
            continue

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc):
            continue

        price = float(hist["Close"].iloc[-1])
        vola = float(hist["Close"].pct_change().rolling(20).std().iloc[-1])

        tp_pct, sl_pct = calc_tp_sl(price, vola, mkt_score)
        tp_price = price * (1 + tp_pct)
        sl_price = price * (1 + sl_pct)

        results.append({
            "ticker": ticker,
            "name": row.get("name", ticker),
            "score": float(sc),
            "price": price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": tp_price,
            "sl_price": sl_price
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ============================================================
# LINE送信（絶対に壊れない最強版）
# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定")
        print(text)
        return

    try:
        r = requests.post(
            WORKER_URL,
            json={"text": text},
            timeout=15
        )
        print("[LINE RESULT]", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] LINE送信失敗:", e)
        print(text)


# ============================================================
# レポート組み立て
# ============================================================
def build_report():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = calc_market_score()
    mkt_score = mkt["score"]
    mkt_comment = mkt["comment"]

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev_used, risk_info = analyze_positions(pos_df)

    results = run_screening(today_date, mkt_score)

    A_list = results[:3]

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推定運用資産ベース: 約{int(total_asset):,}円\n")

    secs = top_sectors_5d()
    if secs:
        lines.append("◆ 今日のTOPセクター（5日騰落率）")
        lines += [f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)]
        lines.append("")

    lines.append("◆ Core候補 Aランク（本命押し目・最大3銘柄）")
    if not A_list:
        lines.append("Aランクなし\n")
    else:
        for r in A_list:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}"
            )
            lines.append(
                f"    ・IN目安: {r['price']:.1f}"
            )
            lines.append(
                f"    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）"
            )
            lines.append(
                f"    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）\n"
            )

    lines.append("📊 ポジション分析")
    lines.append(pos_text)

    return "\n".join(lines)


# ============================================================
# Entry
# ============================================================
def main():
    text = build_report()
    print(text)
    send_line(text)


if __name__ == "__main__":
    main()