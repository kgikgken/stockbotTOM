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


# ============================================
# 基本設定
# ============================================
UNIVERSE_PATH = "universe_jpx.csv"
POSITIONS_PATH = "positions.csv"
WORKER_URL = os.getenv("WORKER_URL")

# 決算フィルタの幅（日）
EARNINGS_EXCLUDE_DAYS = 3


# ============================================
# 日付系
# ============================================
def jst_today_date() -> datetime.date:
    return datetime.now(timezone(timedelta(hours=9))).date()


# ============================================
# イベント（自動拡張予定）
# ============================================
EVENT_CALENDAR: List[Dict[str, str]] = []


def build_event_warnings(today: datetime.date) -> List[str]:
    warns: List[str] = []
    for ev in EVENT_CALENDAR:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
        except Exception:
            continue

        delta = (d - today).days
        if -1 <= delta <= 2:  # 前日〜翌日まで警告
            if delta > 0:
                when = f"{delta}日後"
            elif delta == 0:
                when = "本日"
            else:
                when = "直近"
            warns.append(f"⚠ {ev['label']}（{when}）: ポジションサイズ注意")

    return warns


# ============================================
# Universe 読み込み
# ============================================
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

    # 決算日パース
    if "earnings_date" in df.columns:
        df["earnings_date_parsed"] = pd.to_datetime(
            df["earnings_date"], errors="coerce"
        ).dt.date
    else:
        df["earnings_date_parsed"] = pd.NaT

    return df


def in_earnings_window(row, today) -> bool:
    d = row.get("earnings_date_parsed")
    if d is None or pd.isna(d):
        return False
    try:
        delta = abs((d - today).days)
        return delta <= EARNINGS_EXCLUDE_DAYS
    except:
        return False


# ============================================
# 株価取得
# ============================================
def fetch_history(ticker: str, period="130d"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None


# ============================================
# レバレッジ（最強版）
# ============================================
def calc_target_leverage(mkt_score: int) -> Tuple[float, str]:
    if mkt_score >= 70:
        return 2.0, "攻め（Aランク3フル）"
    if mkt_score >= 60:
        return 1.6, "強め（押し目＋一部ブレイク可）"
    if mkt_score >= 50:
        return 1.3, "標準（押し目のみ）"
    if mkt_score >= 40:
        return 1.0, "守り気味"
    return 0.8, "守り優先"


# ============================================
# IN価格 最強ロジック（ver2）
# ============================================
def calc_best_in_price(hist: pd.DataFrame) -> float:
    close = hist["Close"].astype(float)

    ma20 = close.rolling(20).mean().iloc[-1]
    low20 = close.rolling(20).min().iloc[-1]

    diff = ma20 - low20
    zone_50 = low20 + diff * 0.5
    zone_80 = low20 + diff * 0.8

    candidates = [ma20, zone_50, zone_80, low20]
    candidates = [c for c in candidates if np.isfinite(c)]

    current = close.iloc[-1]
    valid = [c for c in candidates if c < current]

    if valid:
        best = max(valid)
    else:
        best = current * 0.985  # -1.5% fallback

    return round(float(best), 1)


# ============================================
# TP / SL（市場＋ボラ）
# ============================================
def calc_candidate_tp_sl(price, vola20, mkt_score):
    if not np.isfinite(price):
        return 0, 0, price, price

    v = float(vola20) if np.isfinite(vola20) else 0.04

    # ボラ中心
    if v < 0.02:
        tp = 0.08
        sl = -0.03
    elif v > 0.06:
        tp = 0.12
        sl = -0.06
    else:
        tp = 0.10
        sl = -0.04

    # 地合い調整
    if mkt_score >= 70:
        tp += 0.02
    elif mkt_score < 45:
        tp -= 0.02
        sl = max(sl, -0.03)

    tp = float(np.clip(tp, 0.05, 0.18))
    sl = float(np.clip(sl, -0.07, -0.02))

    return tp, sl, price * (1 + tp), price * (1 + sl)


# ============================================
# スクリーニング本体
# ============================================
def run_screening(today, mkt_score) -> List[Dict]:
    df = load_universe()
    if df is None:
        return []

    out = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue

        # 決算フィルタ
        if in_earnings_window(row, today):
            continue

        name = str(row.get("name", ticker))

        hist = fetch_history(ticker)
        if hist is None or len(hist) < 60:
            continue

        sc = score_stock(hist)
        if sc is None or not np.isfinite(sc):
            continue

        close = hist["Close"].astype(float)
        price = close.iloc[-1]

        # ボラ20
        ret = close.pct_change(fill_method=None)
        vola20 = ret.rolling(20).std().iloc[-1]

        # TP/SL
        tp_pct, sl_pct, tp_price, sl_price = calc_candidate_tp_sl(price, vola20, mkt_score)

        # 新ロジック IN価格
        in_price = calc_best_in_price(hist)

        out.append({
            "ticker": ticker,
            "name": name,
            "score": float(sc),
            "price": price,
            "in_price": in_price,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": tp_price,
            "sl_price": sl_price,
        })

    # スコア順
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


# ============================================
# 推奨銘柄選定（最大3）
# ============================================
def pick_top3(results: List[Dict]) -> List[Dict]:
    return results[:3]


# ============================================
# レポート組み立て
# ============================================
def build_core_report(today_str, today_date, mkt, total_asset):
    mkt_score = int(mkt["score"])
    mkt_comment = mkt["comment"]

    lev, lev_label = calc_target_leverage(mkt_score)

    # セクター
    secs = top_sectors_5d()
    if secs:
        sec_text = "\n".join([f"{i+1}. {s[0]} ({s[1]:+.2f}%)" for i, s in enumerate(secs)])
    else:
        sec_text = "算出不可"

    # イベント警告
    warns = build_event_warnings(today_date)

    # スクリーニング
    results = run_screening(today_date, mkt_score)
    picks = pick_top3(results)

    lines = []
    lines.append(f"📅 {today_str} stockbotTOM 日報\n")
    lines.append("◆ 今日の結論")
    lines.append(f"- 地合いスコア: {mkt_score}点")
    lines.append(f"- コメント: {mkt_comment}")
    lines.append(f"- 推奨レバ: 約{lev:.1f}倍（{lev_label}）")
    lines.append(f"- 推定運用資産ベース: 約{int(total_asset):,}円")
    if warns:
        for w in warns:
            lines.append(w)
    lines.append("")

    lines.append("◆ 今日のTOPセクター（5日騰落率）")
    lines.append(sec_text + "\n")

    lines.append("◆ Core候補 Aランク（本命押し目・最大3銘柄）")
    if not picks:
        lines.append("本命候補なし")
    else:
        for r in picks:
            lines.append(
                f"- {r['ticker']} {r['name']}  Score:{r['score']:.1f}  現値:{r['price']:.1f}"
            )
            lines.append(
                f"    ・IN目安: {r['in_price']:.1f}\n"
                f"    ・利確目安: +{r['tp_pct']*100:.1f}%（{r['tp_price']:.1f}）\n"
                f"    ・損切り目安: {r['sl_pct']*100:.1f}%（{r['sl_price']:.1f}）"
            )
            lines.append("")

    return "\n".join(lines)


def build_position_report(today_str, pos_text):
    return f"📊 {today_str} ポジション分析\n\n{pos_text}"


# ============================================
# LINE送信
# ============================================
def send_line(text):
    if not WORKER_URL:
        print("[WARN] WORKER_URL未設定")
        print(text)
        return

    try:
        r = requests.post(WORKER_URL, json={"text": text}, timeout=10)
        print("[LINE RESULT]", r.status_code, r.text)
    except Exception as e:
        print("[ERROR] LINE送信失敗:", e)
        print(text)


# ============================================
# MAIN
# ============================================
def main():
    today_str = jst_today_str()
    today_date = jst_today_date()

    mkt = calc_market_score()

    pos_df = load_positions(POSITIONS_PATH)
    pos_text, total_asset, total_pos, lev, risk = analyze_positions(pos_df)

    core = build_core_report(today_str, today_date, mkt, total_asset)
    pos = build_position_report(today_str, pos_text)

    print(core)
    print("\n" + "=" * 40 + "\n")
    print(pos)

    send_line(core)
    send_line(pos)


if __name__ == "__main__":
    main()