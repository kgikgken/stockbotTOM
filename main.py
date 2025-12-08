from __future__ import annotations
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

from utils.market import enhance_market_score
from utils.sector import top_sectors_5d
from utils.position import load_positions, analyze_positions, compute_positions_rr
from utils.scoring import score_stock
from utils.util import jst_today_str
from utils.rr import compute_tp_sl_rr


# ------------------------------------------------------------
# 基本設定
# ------------------------------------------------------------
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")

SCREENING_TOP_N = 10
MAX_FINAL_STOCKS = 3


# ------------------------------------------------------------
# JST 今日
# ------------------------------------------------------------
def jst_today_date():
    return datetime.now(timezone(timedelta(hours=9))).date()


# ------------------------------------------------------------
# universe 読み込み
# ------------------------------------------------------------
def load_universe(path: str = UNIVERSE_PATH):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "ticker" not in df.columns:
        return None

    df["ticker"] = df["ticker"].astype(str)
    return df


# ------------------------------------------------------------
# 履歴取得
# ------------------------------------------------------------
import yfinance as yf

def fetch_history(ticker: str, period: str = "130d"):
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    return None


# ------------------------------------------------------------
# IN価格（簡易）
# ------------------------------------------------------------
def calc_entry(hist: pd.DataFrame):
    close = hist["Close"].astype(float)
    ma20 = close.rolling(20).mean().iloc[-1]
    return float(round(ma20, 1))


# ------------------------------------------------------------
# スクリーニング
# ------------------------------------------------------------
def run_screening(today, mkt_score: int):
    df = load_universe()
    if df is None:
        return []

    results = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        name = row.get("name", ticker)
        sector = str(row.get("sector", "不明"))

        hist = fetch_history(ticker, "130d")
        if hist is None or len(hist) < 60:
            continue

        base_score = score_stock(hist)
        if base_score is None or base_score < 78:
            continue

        entry = calc_entry(hist)
        price = float(hist["Close"].astype(float).iloc[-1])
        tp_pct, sl_pct, rr = compute_tp_sl_rr(hist, mkt_score)

        # RRフィルタ
        if rr < 1.5:
            continue

        tp_price = entry * (1.0 + tp_pct)
        sl_price = entry * (1.0 + sl_pct)

        gap = abs(price - entry) / price if price > 0 else 1.0
        entry_type = "today" if gap <= 0.01 else "soon"

        results.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "score": float(base_score),
                "price": price,
                "entry": entry,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "rr": rr,
                "entry_type": entry_type,
            }
        )

    results.sort(key=lambda x: x["rr"], reverse=True)
    return results[:MAX_FINAL_STOCKS]


# ------------------------------------------------------------
# 建玉最大
# ------------------------------------------------------------
def recommend_leverage(mkt_score: int):
    if mkt_score >= 70:
        return 1.8, "強め"
    if mkt_score >= 60:
        return 1.5, "やや強め"
    if mkt_score >= 50:
        return 1.3, "標準"
    if mkt_score >= 40:
        return 1.1, "守り"
    return 1.0, "最小"


def calc_max_position(asset: float, lev: float):
    return int(round(asset * lev))


# ------------------------------------------------------------
# 地合いコメント
# ------------------------------------------------------------
def bias_comment(score: int):
    if score >= 70:
        return "強気（押し目＋ブレイク）"
    if score >= 60:
        return "やや強気（押し目主体）"
    if score >= 50:
        return "中立（押し目）"
    if score >= 40:
        return "弱気（サイズ抑え）"
    return "非常に弱い"


# ------------------------------------------------------------
# レポート
# ------------------------------------------------------------
def build_report(today_str, today, mkt, total_asset, pos_text):
    mkt_score = int(mkt.get("score", 50))
    rec_lev, _ = recommend_leverage(mkt_score)
    asset = total_asset if total_asset > 0 else 2_000_000.0
    max_pos = calc_max_position(asset, rec_lev)

    secs = top_sectors_5d()
    sec_text = "\n".join(
        [f"{i+1}. {name} ({chg:+.2f}%)" for i, (name, chg) in enumerate(secs[:5])]
    )

    core = run_screening(today, mkt_score)
    today_list = [c for c in core if c["entry_type"] == "today"]
    soon_list = [c for c in core if c["entry_type"] == "soon"]

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合い: {mkt_score}点 ({bias_comment(mkt_score)})")
    lines.append(f"- レバ: {rec_lev:.1f}倍")
    lines.append(f"- MAX建玉: 約{max_pos:,}円\n")

    lines.append("📈 セクター（5日）")
    lines.append(sec_text + "\n")

    lines.append("🏆 Core候補（最大3銘柄）")
    if not core:
        lines.append("該当なし\n")
    else:
        for c in core:
            lines.append(f"- {c['ticker']} {c['name']} [{c['sector']}]")
            lines.append(f"Score:{c['score']:.1f} RR:{c['rr']:.2f}R")
            lines.append(f"IN:{c['entry']:.1f} TP:+{c['tp_pct']*100:.1f}% SL:{c['sl_pct']*100:.1f}%\n")

    lines.append("📊 ポジション")
    lines.append(pos_text.strip())
    return "\n".join(lines)


# ------------------------------------------------------------
# LINE送信（分割）
# ------------------------------------------------------------
def send_line(text: str):
    if not WORKER_URL:
        print(text)
        return

    size = 3900
    parts = [text[i:i+size] for i in range(0, len(text), size)]
    for p in parts:
        try:
            r = requests.post(WORKER_URL, json={"text": p}, timeout=10)
            print("[LINE]", r.status_code)
        except Exception as e:
            print("[LINE ERR]", e)
            print(p)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    today_str = jst_today_str()
    today = jst_today_date()

    mkt = enhance_market_score()
    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, _, _, _ = analyze_positions(pos_df)

    report = build_report(today_str, today, mkt, total_asset, pos_text)
    print(report)
    send_line(report)


if __name__ == "__main__":
    main()