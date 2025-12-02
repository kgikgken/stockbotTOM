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

EARNINGS_EXCLUDE_DAYS = 3


# ===============================
# 地合い & 日付
# ===============================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ===============================
# 決算フィルタ
# ===============================
def load_universe(path=UNIVERSE_PATH):
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


def in_earnings_window(row, today):
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        return abs((d - today).days) <= EARNINGS_EXCLUDE_DAYS
    except:
        return False


# ===============================
# Data fetch
# ===============================
def fetch_history(ticker, period="130d"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# ===============================
# レバレッジ
# ===============================
def calc_target_leverage(mkt_score: int):
    if mkt_score >= 70:
        return 2.0, "攻め"
    if mkt_score >= 50:
        return 1.3, "標準"
    if mkt_score >= 40:
        return 1.0, "守り"
    return 0.8, "超守り"


# ===============================
# 利確 / 損切り / IN価格
# ===============================
def calc_candidate_tp_sl(price, vola20, mkt_score):
    if vola20 is None or not np.isfinite(vola20):
        vola20 = 0.04

    if vola20 < 0.02:
        tp = 0.08
        sl = -0.03
    elif vola20 > 0.06:
        tp = 0.12
        sl = -0.06
    else:
        tp = 0.10
        sl = -0.04

    if mkt_score >= 70:
        tp += 0.02
    elif mkt_score < 45:
        tp -= 0.02
        sl = max(sl, -0.03)

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))

    return tp, sl


def calc_in_price(hist):
    c = hist["Close"]
    v = hist["Volume"]
    ma5 = c.rolling(5).mean()
    v_ma20 = v.rolling(20).mean()

    for i in range(2, 12):
        if i >= len(ma5):
            break
        if c.iloc[-i] < ma5.iloc[-i] and v.iloc[-i] < v_ma20.iloc[-i]:
            return round(float(c.iloc[-i]), 1)

    return round(float(c.iloc[-1] * 0.90), 1)


# ===============================
# スクリーニング
# ===============================
def run_screening(today, mkt_score):
    df = load_universe()
    if df is None:
        return []

    results = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        if not ticker:
            continue

        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc):
            continue

        c = hist["Close"].astype(float)
        price = float(c.iloc[-1])

        ret = c.pct_change()
        vola20 = ret.rolling(20).std().iloc[-1]

        in_price = calc_in_price(hist)
        tp_pct, sl_pct = calc_candidate_tp_sl(price, vola20, mkt_score)

        results.append({
            "ticker": ticker,
            "name": name,
            "score": float(sc),
            "price": price,
            "in_price": in_price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": round(in_price * (1 + tp_pct), 1),
            "sl_price": round(in_price * (1 + sl_pct), 1),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ===============================
# 分割送信
# ===============================
def send_line_chunks(text: str):
    if not WORKER_URL:
        print("[ERROR] WORKER_URL not set")
        print(text)
        return

    chunk = 3900
    parts = [text[i:i+chunk] for i in range(0, len(text), chunk)]

    for idx, p in enumerate(parts):
        try:
            r = requests.post(WORKER_URL, json={"text": p}, timeout=10)
            print(f"[LINE PART {idx+1}/{len(parts)}]", r.status_code)
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)


# ===============================
# レポート
# ===============================
def build_report():
    today = jst_today_date()
    today_str = jst_today_str()

    mkt = calc_market_score()
    mkt_score = mkt["score"]
    mkt_comment = mkt["comment"]

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, *_ = analyze_positions(pos_df)

    secs = top_sectors_5d()
    sec_text = "\n".join([
        f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)
    ]) if secs else "データ不足"

    results = run_screening(today, mkt_score)
    A = results[:3]

    # ===== パート① 結論 =====
    p1 = []
    p1.append(f"📅 {today_str} stockbotTOM 日報\n")
    p1.append("◆ 今日の結論")
    p1.append(f"- 地合いスコア: {mkt_score}点")
    p1.append(f"- コメント: {mkt_comment}")
    p1.append(f"- 推定運用資産ベース: 約{int(total_asset):,}円")
    part1 = "\n".join(p1)

    # ===== パート② セクター =====
    part2 = f"◆ 今日のTOPセクター（5日騰落率）\n{sec_text}"

    # ===== パート③ Aランク =====
    p3 = []
    p3.append("◆ Core候補 Aランク（本命押し目・最大3銘柄）")
    for r in A:
        p3.append(
            f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f} 現値:{r['price']:.1f}\n"
            f"    ・IN目安: {r['in_price']}\n"
            f"    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']}）\n"
            f"    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']}）"
        )
    part3 = "\n".join(p3)

    # ===== パート④ ポジション =====
    part4 = f"📊 {today_str} ポジション分析\n{pos_text}"

    return [part1, part2, part3, part4]


# ===============================
# Main
# ===============================
def main():
    parts = build_report()
    for p in parts:
        send_line_chunks(p)


if __name__ == "__main__":
    main()