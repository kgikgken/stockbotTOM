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
# 日付
# ============================================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ============================================================
# Universe 読込
# ============================================================
def load_universe(path: str = UNIVERSE_PATH):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except:
        return None

    if "ticker" not in df.columns:
        return None

    df["ticker"] = df["ticker"].astype(str)

    # earnings_date の安全パース
    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row: pd.Series, today: datetime.date) -> bool:
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
def fetch_history(ticker: str, period="130d"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# ============================================================
# 最強IN価格ロジック
# ============================================================
def calc_best_entry_price(hist: pd.DataFrame) -> float:
    """
    最強トレーダー仕様：
    IN = 20MA / 10〜20日スイング安値 / 50MA / 現値 のうち
         「最も高い＝押しすぎないサポ」を採用
    ※トレンド崩壊を避ける
    """
    close = hist["Close"].astype(float)

    # 20MA
    ma20 = float(close.rolling(20).mean().iloc[-1])

    # 50MA
    ma50 = float(close.rolling(50).mean().iloc[-1])

    # 直近10〜20日の最安値
    recent_low = float(close.rolling(20).min().iloc[-1])

    # 現値
    last = float(close.iloc[-1])

    # 候補（NaN除外）
    cands = [x for x in [ma20, recent_low, ma50, last] if np.isfinite(x) and x > 0]

    if not cands:
        return last

    # 押しすぎず、トレンド崩壊しない「最も高いサポート」を採用
    return max(cands)


# ============================================================
# TP / SL
# ============================================================
def calc_candidate_tp_sl(price, vola20, mkt_score):
    if vola20 is None or not np.isfinite(vola20):
        vola20 = 0.04

    # ボラ基準
    if vola20 < 0.02:
        tp = 0.08; sl = -0.03
    elif vola20 > 0.06:
        tp = 0.12; sl = -0.06
    else:
        tp = 0.10; sl = -0.04

    # 地合い補正
    if mkt_score >= 70:
        tp += 0.02
    elif mkt_score < 45:
        tp -= 0.02
        sl = max(sl, -0.03)

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))

    return tp, sl, price * (1 + tp), price * (1 + sl)


# ============================================================
# レバ設定（固定 or 自動）
# ============================================================
def calc_target_leverage(mkt_score):
    return 1.3, "標準（押し目限定）"   # ← 前回仕様どおり固定


# ============================================================
# スクリーニング
# ============================================================
def run_screening(today, mkt_score, total_asset):
    df = load_universe()
    if df is None:
        return [], []

    A, B = [], []

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

        score = score_stock(hist)
        if score is None or not np.isfinite(score):
            continue

        close = hist["Close"].astype(float)
        price = float(close.iloc[-1])

        vola20 = float(close.pct_change().rolling(20).std().iloc[-1])

        # 最強 IN ロジック
        entry = calc_best_entry_price(hist)

        tp_pct, sl_pct, tp_price, sl_price = calc_candidate_tp_sl(price, vola20, mkt_score)

        info = {
            "ticker": ticker,
            "name": name,
            "score": float(score),
            "price": price,
            "entry": entry,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": tp_price,
            "sl_price": sl_price,
        }

        if score >= 85:
            A.append(info)
        elif score >= 75:
            B.append(info)

    A.sort(key=lambda x: x["score"], reverse=True)
    B.sort(key=lambda x: x["score"], reverse=True)
    return A, B


def select_primary_targets(A, B, max_names=3):
    if len(A) >= max_names:
        return A[:max_names], []
    if len(A) > 0:
        need = max_names - len(A)
        return A + B[:need], B[need:]
    return B[:max_names], B[max_names:]


# ============================================================
# レポート構築
# ============================================================
def build_core_report(today_str, today_date, mkt, total_asset):
    mkt_score = int(mkt["score"])
    mkt_comment = mkt["comment"]

    target_lev, lev_label = calc_target_leverage(mkt_score)

    secs = top_sectors_5d()
    if secs:
        sec_text = "\n".join(
            [f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)]
        )
    else:
        sec_text = "算出不可"

    A, B = run_screening(today_date, mkt_score, total_asset)
    primary, _ = select_primary_targets(A, B)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推定運用資産ベース: 約{int(total_asset):,}円\n")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sec_text + "\n")

    lines.append("◆ Core候補 Aランク（本命押し目・最大3銘柄）")
    if not primary:
        lines.append("本命Aランク候補なし。\n")
    else:
        for r in primary:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}"
            )
            lines.append(
                f"    ・IN目安: {r['entry']:.1f}\n"
                f"    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）\n"
                f"    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）\n"
            )

    return "\n".join(lines)


def build_position_report(today_str, pos_text):
    return (
        f"📊 {today_str} ポジション分析\n\n"
        f"◆ ポジションサマリ\n{pos_text.strip()}"
    )


# ============================================================
# LINE送信
# ============================================================
def send_line(text):
    if not WORKER_URL:
        print("[WARN] WORKER_URL 未設定")
        print(text)
        return
    try:
        r = requests.post(WORKER_URL, json={"text": text}, timeout=10)
        print("[LINE RESULT]", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] LINE送信失敗:", e)
        print(text)


# ============================================================
# Entry
# ============================================================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = calc_market_score()

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk_info = analyze_positions(pos_df)

    core = build_core_report(today_str, today_date, mkt, total_asset)
    pos_r = build_position_report(today_str, pos_text)

    print(core)
    print("\n" + "="*40 + "\n")
    print(pos_r)

    send_line(core)
    send_line(pos_r)


if __name__ == "__main__":
    main()