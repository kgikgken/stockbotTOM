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
# データ取得（安全版）
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
# スクリーニング実行
# ============================================================
def calc_inout_guidance(hist):
    """IN目安・利確・損切りのガイドライン（簡易版）"""
    close = hist["Close"]
    last = float(close.iloc[-1])
    ma20 = close.rolling(20).mean().iloc[-1]

    # ボラ
    vola20 = close.pct_change().rolling(20).std().iloc[-1]
    vola = float(vola20) if np.isfinite(vola20) else 0.02

    # IN目安
    if last < ma20:
        in_comment = "押し目圏（IN候補）"
    else:
        in_comment = "上昇中（INは慎重）"

    # 利確 +2〜3σ
    tp = last * (1 + 2 * vola)
    # 損切り -2σ
    sl = last * (1 - 2 * vola)

    return in_comment, tp, sl


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
        if sc is None:
            continue

        price = float(hist["Close"].iloc[-1])

        # IN目安・利確・損切り
        in_comment, tp, sl = calc_inout_guidance(hist)

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": sc,
            "price": price,
            "in_comment": in_comment,
            "tp": tp,
            "sl": sl,
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
    pos_text, total_asset, total_pos, lev, risk_info = analyze_positions(pos_df)

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

    # ---- Aランク ----
    lines.append("◆ Core候補 Aランク（本命押し目）")
    if not A_list:
        lines.append("本命Aランクなし。")
    else:
        for r in A_list:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']}  現値:{r['price']:.1f}"
            )
            lines.append(
                f"    IN目安:{r['in_comment']} / 利確:{r['tp']:.1f} / 損切:{r['sl']:.1f}"
            )
    lines.append("")

    # ---- Bランク ----
    lines.append("◆ Core候補 Bランク（押し目候補）")
    if not B_list:
        lines.append("Bランク候補なし。")
    else:
        for r in B_list:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']}  現値:{r['price']:.1f}"
            )
            lines.append(
                f"    IN目安:{r['in_comment']} / 利確:{r['tp']:.1f} / 損切:{r['sl']:.1f}"
            )
    lines.append("")

    # ---- ポジション ----
    lines.append("◆ ポジション分析")
    lines.append(pos_text)
    lines.append("")
    lines.append("◆ 推奨ポジションリスク")
    lines.append(risk_info)

    return "\n".join(lines)


# ============================================================
# LINE送信（長文分割＋エラー検知付き）
# ============================================================
def send_line(text: str) -> None:
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定。以下を標準出力のみ:")
        print(text)
        return

    max_len = 4000
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)] or ["(empty)"]

    for idx, part in enumerate(chunks, start=1):
        try:
            print(f"[INFO] LINE送信 {idx}/{len(chunks)} 文字数={len(part)}")
            r = requests.post(WORKER_URL, json={"text": part}, timeout=20)
            print("[LINE RESULT]", r.status_code, r.text[:200])
            if r.status_code != 200:
                raise RuntimeError(f"Worker error: {r.status_code} {r.text}")
        except Exception as e:
            print("[ERROR] LINE送信に失敗:", repr(e))
            raise


# ============================================================
# Entry
# ============================================================
def main():
    text = build_report()
    print(text)
    send_line(text)


if __name__ == "__main__":
    main()