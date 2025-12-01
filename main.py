from __future__ import annotations
import os
from typing import List, Dict, Tuple

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

# 表示する銘柄数の上限
MAX_A = 3          # Aランク最大表示数
MAX_TOTAL = 5      # A+B 合計最大表示数


# ============================================================
# データ取得（安全版）
# ============================================================
def fetch_history(ticker: str, period: str = "130d"):
    """個別銘柄の株価履歴を安全に取得"""
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def calc_vola20(close: pd.Series) -> float:
    """20日ボラ（シンプル版）"""
    if close is None or len(close) < 21:
        return float("nan")
    # pandas の将来変更に備えて fill_method=None を明示
    ret = close.pct_change(fill_method=None)
    return float(ret.rolling(20).std().iloc[-1])


def classify_vola(vola: float) -> str:
    """ボラティリティのざっくり分類"""
    if not np.isfinite(vola):
        return "mid"
    if vola < 0.02:
        return "low"
    if vola > 0.06:
        return "high"
    return "mid"


def calc_tp_sl_for_screen(price: float, vola: float, market_score: int) -> Tuple[float, float, float, float]:
    """
    スクリーニング用の利確/損切り目安を計算
    戻り値: (tp_pct, sl_pct, tp_price, sl_price)
    """
    vola_class = classify_vola(vola)

    # --- 利確 ---
    tp = 0.08  # ベース 8%
    if vola_class == "low":
        tp -= 0.01
    elif vola_class == "high":
        tp += 0.02

    if market_score >= 70:
        tp += 0.02
    elif market_score <= 40:
        tp -= 0.02

    tp = float(np.clip(tp, 0.06, 0.15))

    # --- 損切り ---
    sl = -0.04  # ベース -4%
    if vola_class == "low":
        sl = -0.03
    elif vola_class == "high":
        sl = -0.05

    if market_score >= 70:
        sl -= 0.005
    elif market_score <= 40:
        sl += 0.005

    sl = float(np.clip(sl, -0.07, -0.02))

    tp_price = price * (1.0 + tp)
    sl_price = price * (1.0 + sl)
    return tp, sl, tp_price, sl_price


# ============================================================
# スクリーニング実行
# ============================================================
def run_screening(market_score: int):
    """
    ユニバース全体をスコアリングして
    A/B ランクの中から「本当に組みたい」3〜5銘柄だけに絞る
    """
    try:
        uni = pd.read_csv(UNIVERSE_PATH)
    except Exception:
        return [], []

    if "ticker" not in uni.columns:
        return [], []

    candidates_a: List[Dict] = []
    candidates_b: List[Dict] = []

    for _, row in uni.iterrows():
        ticker = str(row["ticker"])
        name = str(row.get("name", ticker))
        sector = str(row.get("sector", "不明"))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        score = score_stock(hist)
        if score is None or not np.isfinite(score):
            continue

        # そもそも弱い銘柄は除外（70未満は無視）
        if score < 70:
            continue

        close = hist["Close"].astype(float)
        price = float(close.iloc[-1])
        vola20 = calc_vola20(close)
        tp_pct, sl_pct, tp_price, sl_price = calc_tp_sl_for_screen(price, vola20, market_score)

        info = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "score": float(score),
            "price": price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }

        # スコア帯で A/B ランク分け（A は少しだけ厳しめ）
        if score >= 85:
            candidates_a.append(info)
        else:
            candidates_b.append(info)

    # スコア順にソート
    candidates_a.sort(key=lambda x: x["score"], reverse=True)
    candidates_b.sort(key=lambda x: x["score"], reverse=True)

    # A から最大 MAX_A
    a_list = candidates_a[:MAX_A]

    # 残り枠を B で埋める（合計 MAX_TOTAL まで）
    remain = max(0, MAX_TOTAL - len(a_list))
    b_list = candidates_b[:remain]

    return a_list, b_list


# ============================================================
# レポート生成
# ============================================================
def build_report():
    today = jst_today_str()

    # ---- 地合い ----
    mkt = calc_market_score()
    mkt_score = int(mkt.get("score", 50))
    mkt_comment = str(mkt.get("comment", "中立"))

    # ---- セクター ----
    secs = top_sectors_5d()
    if secs:
        sector_lines = [f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)]
        sector_text = "\n".join(sector_lines)
    else:
        sector_text = "算出不可（データ不足）"

    # ---- スクリーニング ----
    a_list, b_list = run_screening(mkt_score)

    # ---- ポジション ----
    try:
        pos_df = load_positions("positions.csv")
        pos_text, *rest = analyze_positions(pos_df)
        # analyze_positions 側で「◆ ポジション分析」見出し込みのテキストを返している想定
        pos_message = pos_text
    except Exception:
        pos_message = "◆ ポジション分析\nポジション情報の取得に失敗しました。"

    # ===== ここから LINE 用メッセージ組み立て =====
    lines: List[str] = []
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
    if not a_list:
        lines.append("本命Aランクなし。")
    else:
        for r in a_list:
            lines.append(f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}")
            lines.append(f"    ・IN目安: {r['price']:.1f}")
            lines.append(f"    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）")
            lines.append(f"    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）")
    lines.append("")

    # ---- Bランク ----
    lines.append("◆ Core候補 Bランク（押し目候補・ロット控えめ）")
    if not b_list:
        lines.append("Bランク候補なし。")
    else:
        for r in b_list:
            lines.append(f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}")
            lines.append(f"    ・IN目安: {r['price']:.1f}")
            lines.append(f"    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）")
            lines.append(f"    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）")

    screen_message = "\n".join(lines)

    return screen_message, pos_message


# ============================================================
# LINE送信
# ============================================================
def send_line(text: str):
    if not WORKER_URL:
        print("[WARN] WORKER_URL が未設定（print のみ）")
        print(text)
        return

    try:
        r = requests.post(WORKER_URL, json={"text": text}, timeout=15)
        print("[LINE RESULT]", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] LINE送信に失敗:", e)
        print(text)


# ============================================================
# Entry
# ============================================================
def main():
    screen_message, pos_message = build_report()
    # GitHub Actions のログ確認用
    print("==== SCREENING ====")
    print(screen_message)
    print("==== POSITIONS ====")
    print(pos_message)

    # LINE には 2 通に分けて送信（長文対策）
    send_line(screen_message)
    send_line(pos_message)


if __name__ == "__main__":
    main()