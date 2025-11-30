from __future__ import annotations
"""
stockbotTOM / main.py

日本株スイングトレード朝イチスクリーニング & 戦略通知ボット（完全版）
- universe_jpx.csv → 全銘柄ユニバース
- yfinance → 日足を取得
- utils.market → 地合いスコア（Market Score）
- utils.scoring → Core A/B スコア判定
- positions.csv → ポジション分析（資産/損益/レバレッジ）
- Cloudflare Worker 経由でLINEに通知
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from utils.market import calc_market_score
from utils.scoring import score_stock, classify_core

import yfinance as yf
import requests


# ============================================================
# LINE 送信用（Worker）
# ============================================================
WORKER_URL = os.getenv("WORKER_URL")

def send_line_message(text: str):
    if not WORKER_URL:
        print("[ERROR] WORKER_URL が未設定")
        return
    try:
        r = requests.post(WORKER_URL, json={"message": text})
        print("LINE 送信ステータス:", r.status_code)
    except Exception as e:
        print("LINE送信エラー:", e)


# ============================================================
# 銘柄データ取得
# ============================================================
def fetch_price(ticker: str):
    try:
        df = yf.Ticker(ticker).history(period="3mo")
        if df.empty:
            return None
        return df
    except:
        return None


# ============================================================
# ポジション読み込み
# ============================================================
def load_positions() -> pd.DataFrame:
    try:
        return pd.read_csv("positions.csv")
    except:
        return pd.DataFrame(columns=["ticker", "qty", "avg_price"])


# ============================================================
# ポジション分析
# ============================================================
def analyze_positions(df_pos: pd.DataFrame):
    result_lines = []

    total_value = 0
    for _, row in df_pos.iterrows():
        ticker = row["ticker"]
        qty = row["qty"]
        avg = row["avg_price"]

        hist = fetch_price(ticker)
        if hist is None:
            result_lines.append(f"- {ticker}: データ取得失敗（現値不明）")
            continue

        current = float(hist["Close"].iloc[-1])
        pnl_pct = (current - avg) / avg * 100
        pv = current * qty
        total_value += pv

        result_lines.append(
            f"- {ticker}: 現値 {current:.1f} / 取得 {avg:.1f} / 損益 {pnl_pct:+.2f}%"
        )

    # 推定資産
    try:
        with open("data/equity.json", "r") as f:
            equity_data = json.load(f)
            est_equity = equity_data.get("equity", 3000000)
    except:
        est_equity = 3000000

    leverage = total_value / est_equity if est_equity > 0 else 0

    return result_lines, total_value, est_equity, leverage


# ============================================================
# Core 候補スクリーニング
# ============================================================
def run_screening():
    # 銘柄ユニバース
    uni = pd.read_csv("universe_jpx.csv")

    results_A = []
    results_B = []

    for _, row in uni.iterrows():
        ticker = row["ticker"]
        hist = fetch_price(ticker)
        if hist is None:
            continue

        score = score_stock(hist)
        rank = classify_core(score)

        if rank == "A":
            results_A.append((ticker, score))
        elif rank == "B":
            results_B.append((ticker, score))

    # スコア順に並べる
    results_A.sort(key=lambda x: x[1], reverse=True)
    results_B.sort(key=lambda x: x[1], reverse=True)

    return results_A, results_B


# ============================================================
# 日報作成
# ============================================================
def build_report():
    # ---- 地合い ----
    mkt = calc_market_score()
    market_score = mkt["score"]
    market_comment = mkt["comment"]

    # ---- スクリーニング ----
    A_list, B_list = run_screening()

    # ---- ポジション ----
    df_pos = load_positions()
    pos_lines, total_value, est_equity, leverage = analyze_positions(df_pos)

    # ---- レポート文面 ----
    lines = []
    today = (datetime.now(timezone(timedelta(hours=9)))).strftime("%Y-%m-%d")

    lines.append(f"📅 {today} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {market_score}点")
    lines.append(f"- コメント: {market_comment}")
    lines.append(f"- 推定運用資産: {est_equity:,.0f}円\n")

    # ---- Core A ----
    lines.append("◆ Core候補 Aランク（本命押し目）")
    if len(A_list) == 0:
        lines.append("本命Aランクなし。")
    else:
        for t, s in A_list[:10]:
            lines.append(f"{t} : Score {s}")

    # ---- Core B ----
    lines.append("\n◆ Core候補 Bランク（押し目候補）")
    if len(B_list) == 0:
        lines.append("Bランクもなし。")
    else:
        for t, s in B_list[:10]:
            lines.append(f"{t} : Score {s}")

    # ---- ポジション ----
    lines.append("\n◆ ポジション分析")
    lines.append(f"推定ポジション総額: {total_value:,.0f}円（レバ約 {leverage:.2f}倍）")
    if len(pos_lines) == 0:
        lines.append("ポジションなし。")
    else:
        lines.extend(pos_lines)

    return "\n".join(lines)


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    report = build_report()
    print(report)
    send_line_message(report)
