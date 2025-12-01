from __future__ import annotations
import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time

# ====== utils ======
from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.scoring import score_stock, calc_inout_for_stock, calc_vola20
from utils.position import load_positions, analyze_positions
from utils.util import jst_today_str


# ====== 基本設定 ======
UNIVERSE_PATH = "universe_jpx.csv"
POS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")


# ===== LINE送信（1通ずつ / リトライあり） =====
def send_line(text: str):
    if not WORKER_URL:
        print("[WARN] WORKER_URL未設定：printのみ\n", text)
        return

    payload = {"text": text}

    for attempt in range(2):
        try:
            r = requests.post(WORKER_URL, json=payload, timeout=10)
            print(f"[LINE] status={r.status_code}")
            if r.status_code == 200:
                return
        except Exception as e:
            print("[LINE ERROR]", e)
        time.sleep(1)

    print("[FATAL] LINE送信失敗:", text)


# ======== データ取得（安全版）=========
def fetch_history(ticker: str, period="130d"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# ======== スクリーニング ========
def run_screening():
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except:
        return [], []

    if "ticker" not in uni.columns:
        return [], []

    A_list, B_list = [], []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ticker))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        score = score_stock(hist)
        if score is None:
            continue

        price = float(hist["Close"].iloc[-1])
        vola20 = calc_vola20(hist)
        in_rank, tp_pct, sl_pct = calc_inout_for_stock(hist)

        info = {
            "ticker": ticker,
            "name": name,
            "score": score,
            "price": price,
            "in_rank": in_rank,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
        }

        if score >= 80:
            A_list.append(info)
        elif score >= 70:
            B_list.append(info)

    A_list = sorted(A_list, key=lambda x: x["score"], reverse=True)
    B_list = sorted(B_list, key=lambda x: x["score"], reverse=True)

    return A_list, B_list


# ======== レポート生成（各項目ごと）===========
def build_report_parts():

    today = jst_today_str()

    # === 地合い ===
    mkt = calc_market_score()
    mkt_score = mkt["score"]
    mkt_comment = mkt["comment"]

    # === セクター ===
    secs = top_sectors_5d()
    if secs:
        sector_text = "\n".join(
            [f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)]
        )
    else:
        sector_text = "データ不足"

    # === スクリーニング ===
    A_list, B_list = run_screening()

    # === ポジション ===
    pos_df = load_positions(POS_PATH)
    pos_text, total_asset, total_pos, lev, risk_comment = analyze_positions(pos_df)

    # ----PART 1（今日の結論）----
    part1 = f"""
📅 {today} stockbotTOM

◆ 今日の結論
- 地合いスコア: {mkt_score}点
- コメント: {mkt_comment}

◆ 今日のTOPセクター（5日騰落率）
{sector_text}
"""

    # ----PART 2（Core Aランク）----
    if not A_list:
        a_text = "本命Aランクなし。"
    else:
        lines = []
        for r in A_list:
            lines.append(
                f"- {r['ticker']} {r['name']} Score:{r['score']} 現値:{r['price']:.1f}\n"
                f"  IN:{r['in_rank']}  TP:+{r['tp_pct']:.1f}%  SL:-{r['sl_pct']:.1f}%"
            )
        a_text = "\n".join(lines)

    part2 = f"""
◆ Core候補 Aランク（本命押し目）
{a_text}
"""

    # ----PART 3（Core Bランク）----
    if not B_list:
        b_text = "Bランク候補なし。"
    else:
        lines = []
        for r in B_list:
            lines.append(
                f"- {r['ticker']} {r['name']} Score:{r['score']} 現値:{r['price']:.1f}\n"
                f"  IN:{r['in_rank']}  TP:+{r['tp_pct']:.1f}%  SL:-{r['sl_pct']:.1f}%"
            )
        b_text = "\n".join(lines)

    part3 = f"""
◆ Core候補 Bランク（押し目候補）
{b_text}
"""

    # ----PART 4（ポジション分析）----
    part4 = f"""
◆ ポジション分析
{pos_text}

推定運用資産: {total_asset:,.0f}円
推定ポジション総額: {total_pos:,.0f}円（レバ約 {lev:.2f}倍）
{risk_comment}
"""

    return part1.strip(), part2.strip(), part3.strip(), part4.strip()


# ======== entry =========
def main():
    parts = build_report_parts()

    print("[INFO] sending PART1")
    send_line(parts[0])

    print("[INFO] sending PART2")
    send_line(parts[1])

    print("[INFO] sending PART3")
    send_line(parts[2])

    print("[INFO] sending PART4")
    send_line(parts[3])


if __name__ == "__main__":
    main()