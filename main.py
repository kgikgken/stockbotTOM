from __future__ import annotations
import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests

# === utils ===
from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.util import jst_today_str


UNIVERSE_PATH = "universe_jpx.csv"
WORKER_URL = os.getenv("WORKER_URL")


# ============================================================
# 安全版：株価データ取得
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
# 銘柄スクリーニング（A/B）
# ============================================================
def run_screening():
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception as e:
        print("[ERR] universe load:", e)
        return [], []

    if "ticker" not in uni.columns:
        return [], []

    A_list = []
    B_list = []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", ""))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        score = score_stock(hist)
        if score is None or np.isnan(score):
            continue

        price = float(hist["Close"].iloc[-1])

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": float(score),
            "price": price,
        }

        if score >= 80:
            A_list.append(info)
        elif score >= 70:
            B_list.append(info)

    A_list = sorted(A_list, key=lambda x: x["score"], reverse=True)
    B_list = sorted(B_list, key=lambda x: x["score"], reverse=True)
    return A_list, B_list


# ============================================================
# レポート生成
# ============================================================
def build_report():
    today = jst_today_str()

    # ---- 地合い ----
    mkt = calc_market_score()
    mkt_score = mkt["score"]
    mkt_comment = mkt["comment"]

    # ---- セクターTOP ----
    secs = top_sectors_5d()
    if secs:
        sector_text = "\n".join([f"{i+1}. {name} ({pct:+.2f}%)" for i, (name, pct) in enumerate(secs)])
    else:
        sector_text = "算出不可（データ不足）"

    # ---- スクリーニング ----
    A_list, B_list = run_screening()

    # ---- ポジション ----
    pos_df = load_positions()
    pos_text, total_asset, total_pos, lev = analyze_positions(pos_df)

    # ---- assemble ----
    lines = []
    lines.append(f"📅 {today} stockbotTOM 日報\n")

    # === 地合い ===
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append("")

    # === セクター ===
    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text)
    lines.append("")

    # === Aランク ===
    lines.append("◆ Core候補 Aランク（本命押し目）")
    if not A_list:
        lines.append("本命Aランクなし。")
    else:
        for r in A_list:
            lines.append(f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}")
    lines.append("")

    # === Bランク ===
    lines.append("◆ Core候補 Bランク（押し目候補）")
    if not B_list:
        lines.append("Bランク候補なし。")
    else:
        for r in B_list:
            lines.append(f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}")
    lines.append("")

    # === ポジション ===
    lines.append("◆ ポジション分析")
    lines.append(pos_text)
    lines.append(f"- 推定運用資産: {total_asset:,}円")
    lines.append(f"- 推定ポジション総額: {total_pos:,}円（レバ約 {lev:.2f}倍）")

    return "\n".join(lines)


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定 → printのみ")
        print(text)
        return

    try:
        r = requests.post(WORKER_URL, json={"text": text}, timeout=10)
        print("[LINE RESULT]", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] LINE送信に失敗:", e)


# ============================================================
# MAIN
# ============================================================
def main():
    text = build_report()
    print(text)
    send_line(text)


if __name__ == "__main__":
    main()