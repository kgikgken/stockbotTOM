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

# 決算日フィルタ
EARNINGS_EXCLUDE_DAYS = 3


# ================================
# JST
# ================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ================================
# Universe 読み込み
# ================================
def load_universe(path=UNIVERSE_PATH):
    try:
        df = pd.read_csv(path)
    except Exception:
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


# ================================
# 株価取得
# ================================
def fetch_history(ticker: str, period="130d"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# ================================
# IN 価格ロジック
# ================================
def calc_in_price(hist: pd.DataFrame) -> float:
    close = hist["Close"].astype(float)
    ma = close.rolling(20).mean().iloc[-1]

    last = close.iloc[-1]
    vol20 = close.pct_change().rolling(20).std().iloc[-1]
    if not np.isfinite(vol20):
        vol20 = 0.02

    scaled = max(0.02, min(0.06, vol20))

    in_price = float(ma * (1 - scaled))

    # 安全クリップ（現値±20%の範囲）
    lower = last * 0.8
    upper = last * 1.2
    in_price = float(np.clip(in_price, lower, upper))

    return round(in_price, 1)


# ================================
# TP / SL
# ================================
def calc_tp_sl(price: float):
    tp_pct = 0.10
    sl_pct = -0.04
    return (
        tp_pct,
        sl_pct,
        price * (1 + tp_pct),
        price * (1 + sl_pct),
    )


# ================================
# スクリーニング
# ================================
def run_screening(today, total_asset):
    df = load_universe()
    if df is None:
        return []

    results = []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        if not ticker:
            continue

        if in_earnings_window(row, today):
            continue

        name = row.get("name", ticker)
        sector = row.get("sector", row.get("industry_big", "不明"))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        score = score_stock(hist)
        if score is None or not np.isfinite(score):
            continue

        close = hist["Close"].astype(float)
        price = float(close.iloc[-1])

        # IN
        in_price = calc_in_price(hist)

        # TP / SL
        tp_pct, sl_pct, tp_price, sl_price = calc_tp_sl(price)

        results.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "score": float(score),
                "price": price,
                "in": in_price,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def pick_top3(results: List[Dict]):
    return results[:3]


# ================================
# レポート生成
# ================================
def build_core_report(today_str, today_date, total_asset):
    # 市場
    mkt = calc_market_score()
    mkt_score = mkt["score"]
    mkt_comment = mkt["comment"]

    # スクリーニング
    results = run_screening(today_date, total_asset)
    A_list = pick_top3(results)

    # セクター
    secs = top_sectors_5d()
    if secs:
        sec_text = "\n".join([f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)])
    else:
        sec_text = "算出不可（データ不足）"

    # 建て玉最大金額
    max_pos = int(total_asset * 1.3)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推定運用資産ベース: 約{total_asset:,}円\n")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sec_text + "\n")

    lines.append("◆ 今日のイベント・警戒情報")
    lines.append("- 特筆すべきイベントなし（通常モード）\n")

    lines.append("◆ Core候補 Aランク（本命押し目・最大3銘柄）")
    if not A_list:
        lines.append("本命Aランクなし")
    else:
        for r in A_list:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f} 現値:{r['price']:.1f}"
            )
            lines.append(f"    ・IN目安: {r['in']:.1f}")
            lines.append(f"    ・利確目安: +10.0%（{r['tp_price']:.1f}）")
            lines.append(f"    ・損切り目安: -4.0%（{r['sl_price']:.1f}）")
            lines.append("")

    lines.append("◆ 本日の建て玉最大金額")
    lines.append("- 推奨レバ: 1.3倍")
    lines.append(f"- 今日のMAX建て玉: 約{max_pos:,}円")

    return "\n".join(lines)


def build_position_report(today_str, pos_text):
    lines = []
    lines.append(f"📊 {today_str} ポジション分析\n")
    lines.append("◆ ポジションサマリ")
    lines.append(pos_text)
    return "\n".join(lines)


# ================================
# LINE 送信（強化版）
# ================================
def send_line(text: str):
    """確実に届くよう 1000 文字で分割送信"""
    if not WORKER_URL:
        print(text)
        return

    chunk = 1000
    parts = [text[i:i + chunk] for i in range(0, len(text), chunk)]

    for p in parts:
        try:
            r = requests.post(WORKER_URL, json={"text": p}, timeout=10)
            print("[LINE RESULT]", r.status_code, r.text)
        except Exception as e:
            print("[ERROR] LINE送信失敗:", e)


# ================================
# Entry
# ================================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    # ポジション
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk_info = analyze_positions(pos_df)

    # Core
    core_report = build_core_report(
        today_str=today_str,
        today_date=today_date,
        total_asset=total_asset,
    )

    # ポジション
    pos_report = build_position_report(today_str, pos_text)

    # 送信
    send_line(core_report)
    send_line(pos_report)


if __name__ == "__main__":
    main()