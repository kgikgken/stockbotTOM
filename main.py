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


# ============================================================
# 日付関連
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ============================================================
# Universe 読み込み
# ============================================================
def load_universe(path=UNIVERSE_PATH):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return None

    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = None

    return df


def in_earnings_window(row, today):
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        return abs((d - today).days) <= EARNINGS_EXCLUDE_DAYS
    except:
        return False


# ============================================================
# 株価取得
# ============================================================
def fetch_history(ticker, period="130d"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# ============================================================
# IN価格（最強押し目ロジック）
# ============================================================
def calc_in_price(price: float, hist: pd.DataFrame) -> float:
    """
    世界最高トレーダー仕様 IN目安（押し目の最強版）
    ① MA20以下 → MA20
    ② MA20の角度が上昇 → MA20 + α
    ③ ボラが大きすぎる時のみ安全側
    """
    close = hist["Close"].astype(float)
    ma20 = close.rolling(20).mean().iloc[-1]
    slope = ma20 - close.rolling(20).mean().iloc[-5]  # MA20の傾き
    vola = close.pct_change().rolling(20).std().iloc[-1]

    # ボラ高すぎ → 深追い禁止
    if vola > 0.06:
        return round(price * 0.97, 1)

    # MA20下抜け → MA20で押し目拾う
    if price <= ma20:
        return round(ma20, 1)

    # MA20が上向き → 少し上で拾う（強気相場）
    if slope > 0:
        return round(ma20 * 1.005, 1)

    # 通常：MA20
    return round(ma20, 1)


# ============================================================
# 利確・損切り
# ============================================================
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

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))

    return tp, sl, price * (1 + tp), price * (1 + sl)


# ============================================================
# スクリーニング
# ============================================================
def run_screening(today, mkt_score):
    df = load_universe()
    if df is None:
        return [], []

    A_list = []
    B_list = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))
        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        score = score_stock(hist)
        if score is None or not np.isfinite(score):
            continue

        price = float(hist["Close"].iloc[-1])
        vola20 = float(hist["Close"].pct_change().rolling(20).std().iloc[-1])

        in_price = calc_in_price(price, hist)
        tp_pct, sl_pct, tp_price, sl_price = calc_candidate_tp_sl(price, vola20, mkt_score)

        info = {
            "ticker": ticker,
            "name": name,
            "score": float(score),
            "price": price,
            "in_price": in_price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }

        if score >= 85:
            A_list.append(info)
        elif score >= 75:
            B_list.append(info)

    A_list.sort(key=lambda x: x["score"], reverse=True)
    B_list.sort(key=lambda x: x["score"], reverse=True)
    return A_list, B_list


# ============================================================
# 勝率最優先：Aが0の日は新規INなし
# ============================================================
def select_primary_targets(A_list, B_list, max_names=3):
    if len(A_list) >= max_names:
        return A_list[:max_names], B_list

    if 0 < len(A_list) < max_names:
        need = max_names - len(A_list)
        return A_list + B_list[:need], B_list[need:]

    return [], B_list  # Aが0 → 新規INなし


# ============================================================
# レポート生成
# ============================================================
def send_line(text):
    if not WORKER_URL:
        print(text)
        return

    chunks = [text[i:i+3800] for i in range(0, len(text), 3800)]
    for ch in chunks:
        try:
            r = requests.post(WORKER_URL, json={"text": ch}, timeout=10)
            print("[LINE RESULT]", r.status_code, r.text)
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)


def build_report():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = calc_market_score()
    mkt_score = int(mkt["score"])
    mkt_comment = mkt["comment"]
    est_asset = mkt["asset"]
    lev = mkt["leverage"]

    max_pos = int(est_asset * lev)

    secs = top_sectors_5d()
    sector_text = "\n".join([f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)])

    A_list, B_list = run_screening(today_date, mkt_score)
    primary, rest_B = select_primary_targets(A_list, B_list)

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, _, _, _, _ = analyze_positions(pos_df)

    lines = []

    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推定運用資産ベース: 約{est_asset:,}円\n")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text + "\n")

    lines.append("◆ Core候補 Aランク（本命押し目・最大3銘柄）")
    if not primary:
        lines.append("本命Aランクなし（今日は新規IN禁止）\n")
    else:
        for r in primary:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f} 現値:{r['price']:.1f}\n"
                f"    ・IN目安: {r['in_price']:.1f}\n"
                f"    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）\n"
                f"    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）\n"
            )

    lines.append("◆ 本日の建て玉最大金額")
    lines.append(f"- 推奨レバ: {lev:.1f}倍")
    lines.append(f"- 運用資産ベース: 約{est_asset:,}円")
    lines.append(f"- 今日のMAX建て玉: 約{max_pos:,}円\n")

    lines.append("📊 ポジション分析")
    lines.append(pos_text)

    return "\n".join(lines)


def main():
    text = build_report()
    print(text)
    send_line(text)


if __name__ == "__main__":
    main()