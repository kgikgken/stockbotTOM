from __future__ import annotations
import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests

from utils.market import calc_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions
from utils.scoring import score_stock
from utils.util import jst_today_str


# ============================================================
# 基本設定
# ============================================================
UNIVERSE_PATH = "universe_jpx.csv"
WORKER_URL = os.getenv("WORKER_URL")


# ============================================================
# 安全データ取得
# ============================================================
def fetch_history(ticker: str, period="130d"):
    """yfinance安全ラッパー"""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


# ============================================================
# スクリーニング
# ============================================================
def run_screening():
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except:
        return [], []

    if "ticker" not in uni.columns:
        return [], []

    A_list = []
    B_list = []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        sc = score_stock(hist)
        if sc is None or np.isnan(sc):
            continue

        price = float(hist["Close"].iloc[-1])

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": sc,
            "price": price,
        }

        if sc >= 80:
            A_list.append(info)
        elif sc >= 70:
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

    # ---- セクター ----
    secs = top_sectors_5d()
    if secs:
        sector_text = "\n".join([f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)])
    else:
        sector_text = "算出不可（データ不足）"

    # ---- screening ----
    A_list, B_list = run_screening()

    # ---- ポジ ----
    pos_df = load_positions()
    pos_text, total_asset = analyze_positions(pos_df)

    # ---- assemble ----
    lines = []
    lines.append(f"📅 {today} stockbotTOM 日報\n")

    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append("")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sector_text)
    lines.append("")

    lines.append("◆ Core候補 Aランク（本命押し目）")
    if not A_list:
        lines.append("本命Aランクなし。")
    else:
        for r in A_list:
            lines.append(f"- {r['ticker']} {r['name']}  Score:{r['score']}  現値:{r['price']:.1f}")
    lines.append("")

    lines.append("◆ Core候補 Bランク（押し目候補）")
    if not B_list:
        lines.append("Bランク候補なし。")
    else:
        for r in B_list:
            lines.append(f"- {r['ticker']} {r['name']}  Score:{r['score']}  現値:{r['price']:.1f}")
    lines.append("")

    lines.append("◆ ポジション分析")

    # 🔥 pos_text が list / str / None どれでも安全に処理
    if isinstance(pos_text, list):
        lines.extend(pos_text)
    else:
        lines.append(str(pos_text))

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
        print("[ERROR] LINE送信失敗:", e)


# ============================================================
# Entry
# ============================================================
def main():
    report = build_report()
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()